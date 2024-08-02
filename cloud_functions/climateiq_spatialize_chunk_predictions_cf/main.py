from concurrent import futures
import itertools
import json
import pathlib
import os
from typing import Any, Callable

import flask
import functions_framework
import geopandas as gpd
from google.cloud import firestore_v1
from google.cloud.storage import client as gcs_client
from h3 import h3
import numpy as np
import pandas as pd
from shapely import geometry

INPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-chunk-predictions"
OUTPUT_BUCKET_NAME = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-spatialized-chunk-predictions"
)
GLOBAL_CRS = "EPSG:4326"
# CAUTION: Changing the H3 cell size may require updates to how many/which neighboring
# chunks we process.
H3_LEVEL = 13
STUDY_AREAS_ID = "study_areas"
CHUNKS_ID = "chunks"
# Number of processes to split a Dataframe into when adding h3 data.
NUM_PROCESSES = 8


def _get_object_name(request: flask.Request) -> str:
    req_json = request.get_json(silent=True)
    if req_json is None or "object_name" not in req_json:
        raise ValueError("No object_name provided in request.\n")
    return req_json["object_name"]


# Triggered from a HTTP request.
@functions_framework.http
def spatialize_chunk_predictions(request: flask.Request):
    """This function spatializes model predictions for a single chunk.

    Spatialized model predictions are outputtted to a CSV file in GCS,
    containing H3 indexes along with associated predictions.

    Args:
        request: The HTTP request to this Cloud Function.
    """
    object_name = _get_object_name(request)

    # Extract components from the object name.
    path = pathlib.PurePosixPath(object_name)
    if len(path.parts) != 6:
        raise ValueError(
            f"Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'\n"
            f"Actual name: '{object_name}'"
        )

    id, prediction_type, model_id, study_area_name, scenario_id, chunk_id = path.parts
    try:
        predictions = _read_chunk_predictions(object_name)
        study_area_metadata, chunks_ref = _get_study_area_metadata(study_area_name)
        chunk_metadata = _get_chunk_metadata(chunks_ref, chunk_id)

        spatialized_predictions = _build_spatialized_model_predictions(
            study_area_metadata, chunk_metadata, predictions
        )

        h3_predictions = _calculate_h3_indexes(
            study_area_metadata,
            chunk_metadata,
            spatialized_predictions,
            object_name,
            chunks_ref,
        )
    except ValueError as ve:
        # Any raised ValueErrors are non-retriable so return instead of throwing an
        # exception (which would trigger retries)
        raise ValueError(f"Error for {object_name}: {ve}")

    storage_client = gcs_client.Client()
    bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
    blob = bucket.blob(
        f"{id}/{prediction_type}/{model_id}/{study_area_name}/{scenario_id}/{chunk_id}"
        ".csv"
    )
    with blob.open("w", content_type="text/csv") as fd:
        h3_predictions.to_csv(fd)

    return "Success!"


# TODO: Modify this logic once CNN output schema is confirmed. Also update to
# account for errors and special values.
def _read_chunk_predictions(object_name: str) -> np.ndarray:
    """Reads model predictions for a given chunk from GCS.

    Args:
        object_name: The name of the chunk object to read.

    Returns:
        A 2D array containing the model predictions for the chunk.

    Raises:
        ValueError: If the predictions file format is invalid.
    """
    storage_client = gcs_client.Client()
    bucket = storage_client.bucket(INPUT_BUCKET_NAME)
    blob = bucket.blob(object_name)

    with blob.open() as fd:
        fd_iter = iter(fd)
        line = next(fd_iter, None)
        # The file is expected to contain only one prediction.
        if line is None:
            raise ValueError(f"Predictions file: {object_name} is missing.")

        prediction = json.loads(line)["prediction"]

        if next(fd_iter, None) is not None:
            raise ValueError("Predictions file has too many predictions.")

    return np.array(prediction)


def _read_neighbor_chunk_predictions(
    object_name: str, neighbor_chunk_id: str
) -> np.ndarray:
    """Reads model predictions for a neighbor chunk from GCS.

    Args:
        object_name: The name of the chunk object this cloud function is currently
        processing. Used to construct the object name of the neighbor chunk.
        neighbor_chunk_id: The id of the neighbor chunk object to read.

    Returns:
        A 2D array containing the model predictions for the neighbor chunk.

    Raises:
        ValueError: If the predictions file format is invalid.
    """
    path = pathlib.PurePosixPath(object_name)
    if len(path.parts) != 6:
        raise ValueError(
            f"Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'\n"
            f"Actual name: '{object_name}'"
        )
    *prefix, _ = path.parts
    neighbor_object_name = pathlib.PurePosixPath(*prefix, neighbor_chunk_id)
    return _read_chunk_predictions(str(neighbor_object_name))


def _get_study_area_metadata(
    study_area_name: str,
) -> tuple[dict, Any]:
    """Retrieves metadata for a given study area from Firestore.

    Args:
        study_area_name: The name of the study area to retrieve metadata for.

    Returns:
        A dictionary containing metadata for the study area and a reference to the
        chunks collection in Firestore.

    Raises:
        ValueError: If the study area does not exist or its metadata is
        missing required fields.
    """
    # TODO: Consider refactoring this to use library from climateiq-cnn repo.
    db = firestore_v1.Client()

    study_area_ref = db.collection(STUDY_AREAS_ID).document(study_area_name)
    chunks_ref = study_area_ref.collection(CHUNKS_ID)
    study_area_doc = study_area_ref.get()

    if not study_area_doc.exists:
        raise ValueError(f'Study area "{study_area_name}" does not exist')

    if len(chunks_ref.get()) == 0:
        raise ValueError(f'Study area "{study_area_name}" is missing chunks')

    study_area_metadata = study_area_doc.to_dict()
    if (
        not study_area_metadata
        or "cell_size" not in study_area_metadata
        or "crs" not in study_area_metadata
        or "chunk_x_count" not in study_area_metadata
        or "chunk_y_count" not in study_area_metadata
    ):
        raise ValueError(
            f'Study area "{study_area_name}" is missing one or more required '
            "fields: cell_size, crs, chunk_x_count, chunk_y_count"
        )

    return study_area_metadata, chunks_ref


def _chunk_metadata_fields_valid(chunk_metadata: dict) -> bool:
    """Checks whether all required fields are present in chunk_metadata.

    Args:
        chunk_metadata: A dictionary containing metadata for the chunk.

    Returns:
        A bool indicating whether all required fields exist.
    """
    return (
        "row_count" in chunk_metadata
        and "col_count" in chunk_metadata
        and "x_ll_corner" in chunk_metadata
        and "y_ll_corner" in chunk_metadata
        and "x_index" in chunk_metadata
        and "y_index" in chunk_metadata
    )


def _get_chunk_metadata(chunks_ref: Any, chunk_id: str) -> dict:
    """Retrieves metadata for a specific chunk within a study area.

    Args:
        chunks_ref: A reference to the chunks collection in Firestore
        chunk_id: The id of the chunk to retrieve metadata for.

    Returns:
        A dictionary containing metadata for the chunk.

    Raises:
        ValueError: If the specified chunk does not exist or its metadata is
        missing required fields.
    """
    chunk_metadata = chunks_ref.document(chunk_id).get()

    if not chunk_metadata.exists:
        raise ValueError(f'Chunk "{chunk_id}" does not exist')

    chunk_metadata = chunk_metadata.to_dict()

    if not _chunk_metadata_fields_valid(chunk_metadata):
        raise ValueError(
            f'Chunk "{chunk_id}" is missing one or more required fields: '
            "row_count, col_count, x_ll_corner, y_ll_corner, x_index, y_index"
        )

    return chunk_metadata


def _build_spatialized_model_predictions(
    study_area_metadata: dict, chunk_metadata: dict, predictions: np.ndarray
) -> pd.DataFrame:
    """Builds a DF containing the lat/lon coordinates of each cell's center point.

    Args:
        study_area_metadata: A dictionary containing metadata for the study
        area.
        chunk_metadata: A dictionary containing metadata for the chunk.
        predictions: A 2D array containing the model predictions for the chunk.

    Returns:
        A DataFrame containing the lat/lon coordinates of cell centroids in a
        single chunk along with associated predictions, representing a
        subset of the full study area results.
    """
    rows = np.arange(chunk_metadata["row_count"])
    cols = np.arange(chunk_metadata["col_count"])

    # Calculate cell's center point in the source CRS.
    x_centers = (
        chunk_metadata["x_ll_corner"] + (cols + 0.5) * study_area_metadata["cell_size"]
    )
    y_centers = (
        chunk_metadata["y_ll_corner"] + (rows + 0.5) * study_area_metadata["cell_size"]
    )
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    # Convert 2D meshgrids to 1D arrays.
    x_coords = x_grid.flatten()
    y_coords = y_grid.flatten()

    # Convert coordinates from the source CRS to the global CRS (lat/lon).
    gdf_src_crs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=study_area_metadata["crs"],
    )
    gdf_global_crs = gdf_src_crs.to_crs(GLOBAL_CRS)

    # Reverse prediction rows to align with generated coordinates.
    aligned_predictions = np.flipud(predictions).flatten()

    return pd.DataFrame(
        {
            "lat": gdf_global_crs.geometry.y,
            "lon": gdf_global_crs.geometry.x,
            "prediction": aligned_predictions,
        }
    )


def _get_h3_index_for_cells(lats: np.ndarray, lons: np.ndarray) -> list[str]:
    """Projects cell centroids to H3 indices.

    The lats and lons are paired.

    Args:
        lats: Numpy array of latitudes.
        lons: Numpy array of longitudes.

    Returns:
        A list containing the H3 indices of the projected cell centroids.
    """
    cells = np.column_stack((lats, lons))
    return [h3.geo_to_h3(cell[0], cell[1], H3_LEVEL) for cell in cells]


def _get_h3_index_details(lat: float, lon: float, chunk_boundary: Any) -> dict:
    """Projects the cell centroid to a H3 index and adds H3 details.

    Args:
        cell: A cell row containing the lat and lon of the cell centroid.
        chunk_boundary: A shapely.Polygon representing the chunk.

    Returns:
        A dict containing H3 information for the projected cell centroid.
    """
    h3_index = h3.geo_to_h3(lat, lon, H3_LEVEL)
    centroid_lat, centroid_lon = h3.h3_to_geo(h3_index)
    boundary_xy = geometry.Polygon(h3.h3_to_geo_boundary(h3_index, True))
    is_boundary_cell = not boundary_xy.within(chunk_boundary)

    # Filter out any rows where the projected H3 centroid falls outside of the
    # chunk boundary.
    if not chunk_boundary.contains(geometry.Point(centroid_lon, centroid_lat)):
        h3_index = None
        centroid_lat = None
        centroid_lon = None
        boundary_xy = None
        is_boundary_cell = False

    return {
        "h3_index": h3_index,
        "h3_centroid_lat": centroid_lat,
        "h3_centroid_lon": centroid_lon,
        "h3_boundary": boundary_xy,
        "is_boundary_cell": is_boundary_cell,
    }


def _get_h3_index_details_for_cells(
    lats: np.ndarray, lons: np.ndarray, chunk_boundary: Any
) -> list[dict[str, Any]]:
    """Calls _get_h3_index_details on a set of cells.

    The lats and lons are paired.

    Args:
        lats: Numpy array of latitudes.
        lons: Numpy array of longitudes.

    Returns:
        A list of dicts of H3 information for the projected cell centroids.
    """
    cells = np.column_stack((lats, lons))
    return [_get_h3_index_details(cell[0], cell[1], chunk_boundary) for cell in cells]


def _multiprocess_get_h3(
    spatialized_predictions: pd.DataFrame,
    get_h3_fn: Callable[..., list[Any]],
    *get_h3_fn_extra_args,
) -> list[Any]:
    """Calls get h3 indices/details function on a DataFrame of predictions in parallel.

    Args:
        spatialized_predictions: A Pandas DataFrame of predictions.
        get_h3_fn: A function that gets h3 indices or details. Takes args of:
            * Numpy array of latitudes
            * Numpy array of longitudes
            * Any number of additional args.
        get_h3_fn_extra_args: Extra args to pass into get_h3_fn after the lats and lons.
    Returns:
        List of results for each row. Row order is same as original DataFrame.
    """
    max_rows_per_process = max(
        1, int(len(spatialized_predictions.index) / NUM_PROCESSES)
    )
    subset_futures = []
    with futures.ProcessPoolExecutor() as executor:
        for i in range(0, len(spatialized_predictions.index), max_rows_per_process):
            subset = spatialized_predictions[i : i + max_rows_per_process]
            # Pandas is not threadsafe, so we pass in numpy arrays instead.
            future = executor.submit(
                get_h3_fn,
                subset["lat"].to_numpy(),
                subset["lon"].to_numpy(),
                *get_h3_fn_extra_args,
            )
            subset_futures.append(future)
    futures.wait(subset_futures)
    return list(
        itertools.chain.from_iterable(future.result() for future in subset_futures)
    )


def _get_chunk_boundary(study_area_metadata: dict, chunk_metadata: dict):
    """Calculates the boundary points of the chunk.

    Args:
        study_area_metadata: A dictionary containing metadata for the study
        area.
        chunk_metadata: A dictionary containing metadata for the chunk.

    Returns:
        A shapely.Polygon representing the chunk.
    """
    min_x = chunk_metadata["x_ll_corner"]
    min_y = chunk_metadata["y_ll_corner"]
    max_x = min_x + chunk_metadata["col_count"] * study_area_metadata["cell_size"]
    max_y = min_y + chunk_metadata["row_count"] * study_area_metadata["cell_size"]

    boundary_points = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]

    gdf_src_crs = gpd.GeoDataFrame(
        geometry=[geometry.Polygon(boundary_points)],
        crs=study_area_metadata["crs"],
    )
    gdf_global_crs = gdf_src_crs.to_crs(GLOBAL_CRS)
    return gdf_global_crs.geometry[0]


def _calculate_h3_indexes(
    study_area_metadata: dict,
    chunk_metadata: dict,
    spatialized_predictions: pd.DataFrame,
    object_name: str,
    chunks_ref: Any,
) -> pd.Series:
    """Projects cell centroids to H3 indexes.

    Filters out cells that have projected H3 centroids outside of the current
    chunk and de-dupes duplicate H3 projections by using the max prediction
    value from overlapping cells.

    Args:
        study_area_metadata: A dictionary containing metadata for the study
        area.
        chunk_metadata: A dictionary containing metadata for the chunk.
        spatialized_predictions: A DataFrame containing the lat/lon coordinates of cell
        centroids in a single chunk along with associated predictions
        object_name: The name of the chunk object this cloud function is currently
        processing. Used to construct the GCS object name of the neighbor chunk.
        chunks_ref: A reference to the chunks collection in Firestore, used to
        retrieve metadata of the neighbor chunk.

    Returns:
        A Series containing H3 indexes in a single chunk along with associated
        predictions, representing a subset of the full study area results.

    Raises:
        ValueError: If an expected neighbor chunk does not exist or its metadata is
        missing required fields.
    """
    # Calculate H3 information for each cell.
    chunk_boundary = _get_chunk_boundary(study_area_metadata, chunk_metadata)

    h3_index_details = _multiprocess_get_h3(
        spatialized_predictions, _get_h3_index_details_for_cells, chunk_boundary
    )

    # It's more efficient to add all column values at once instead of iterating row by
    # row.
    spatialized_predictions["h3_index"] = [
        detail["h3_index"] for detail in h3_index_details
    ]
    spatialized_predictions["h3_centroid_lat"] = [
        detail["h3_centroid_lat"] for detail in h3_index_details
    ]
    spatialized_predictions["h3_centroid_lon"] = [
        detail["h3_centroid_lon"] for detail in h3_index_details
    ]
    spatialized_predictions["h3_boundary"] = [
        detail["h3_boundary"] for detail in h3_index_details
    ]
    spatialized_predictions["is_boundary_cell"] = [
        detail["is_boundary_cell"] for detail in h3_index_details
    ]

    spatialized_predictions = spatialized_predictions.dropna(how="any")

    # Extract rows where the projected H3 cell is not fully contained within the chunk
    # so we can aggregate prediction values across chunk boundaries.
    boundary_h3_cells = spatialized_predictions[
        spatialized_predictions["is_boundary_cell"]
    ]["h3_boundary"].unique()

    return _aggregate_h3_predictions(
        study_area_metadata,
        chunk_metadata,
        boundary_h3_cells,
        spatialized_predictions,
        object_name,
        chunks_ref,
    )


def _aggregate_h3_predictions(
    study_area_metadata: dict,
    chunk_metadata: dict,
    boundary_h3_cells: np.ndarray,
    spatialized_predictions: pd.DataFrame,
    object_name: str,
    chunks_ref: Any,
) -> pd.Series:
    """Aggregates predictions for duplicate H3 projections across chunk boundaries.

    Args:
        study_area_metadata: A dictionary containing metadata for the study
        area.
        chunk_metadata: A dictionary containing metadata for the chunk.
        boundary_h3_cells: An array containing the H3 boundaries of cells that
        intersect with the chunk boundary.
        spatialized_predictions: A DataFrame containing H3 indexes and predictions
        for a single chunk. This input DataFrame may contain duplicate H3 indexes.
        object_name: The name of the chunk object this cloud function is currently
        processing. Used to construct the GCS object name of the neighbor chunk.
        chunks_ref: A reference to the chunks collection in Firestore, used to
        retrieve metadata of the neighbor chunk.

    Returns:
        A Series containing H3 indexes in a single chunk along with associated
        predictions, representing a subset of the full study area results.

    Raises:
        ValueError: If an expected neighbor chunk does not exist or its metadata is
        missing required fields.
    """
    # No need to read neighbors if current chunk had no H3 cells that overlap with
    # neighbor chunk boundaries.
    if boundary_h3_cells.size == 0:
        spatialized_predictions = spatialized_predictions.groupby(
            ["h3_index"]
        ).prediction.agg("max")
        return spatialized_predictions

    x = chunk_metadata["x_index"]
    y = chunk_metadata["y_index"]

    # Process direct 8 neighbors of current chunk. Since H3 cells at resolution 13
    # (43.870 m^2) have a smaller area than chunks (2000 m^2), it is guaranteed that any
    # overlapping H3 cells from the current chunk are contained within these neighbors.
    neighbors = {
        (x - 1, y + 1),
        (x, y + 1),
        (x + 1, y + 1),
        (x - 1, y),
        (x + 1, y),
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y - 1),
    }
    for neighbor_x, neighbor_y in neighbors:
        if (
            neighbor_x < 0
            or neighbor_y < 0
            or neighbor_x >= study_area_metadata["chunk_x_count"]
            or neighbor_y >= study_area_metadata["chunk_y_count"]
        ):
            # Chunk is outside the study area boundary.
            continue
        query = (
            chunks_ref.where(
                filter=firestore_v1.base_query.FieldFilter("x_index", "==", neighbor_x)
            )
            .where(
                filter=firestore_v1.base_query.FieldFilter("y_index", "==", neighbor_y)
            )
            .limit(1)
        )
        chunk_docs = query.get()
        if len(chunk_docs) == 0:
            raise ValueError(
                f"Neighbor chunk at index {neighbor_x, neighbor_y} is missing from the "
                "study area."
            )

        chunk_doc = chunk_docs[0]
        neighbor_chunk_id = chunk_doc.id
        neighbor_chunk_metadata = chunk_doc.to_dict()
        if not _chunk_metadata_fields_valid(neighbor_chunk_metadata):
            raise ValueError(
                f"Neighbor chunk at index {neighbor_x, neighbor_y} is missing one or "
                "more required fields: id, row_count, col_count, x_ll_corner,"
                "y_ll_corner, x_index, y_index"
            )

        # Determine if the neighbor chunk intersects with any of the boundary H3 cells.
        neighbor_chunk_boundary = _get_chunk_boundary(
            study_area_metadata, neighbor_chunk_metadata
        )
        intersects = False
        for h3_boundary in boundary_h3_cells:
            if h3_boundary.intersects(neighbor_chunk_boundary):
                intersects = True
                break

        # Project cell centroids of neighbor chunk to H3 indexes and append any
        # duplicate H3 indexes to input spatialized_predictions.
        if intersects:
            neighbor_chunk_predictions = _read_neighbor_chunk_predictions(
                object_name, neighbor_chunk_id
            )
            neighbor_chunk_spatialized_predictions = (
                _build_spatialized_model_predictions(
                    study_area_metadata,
                    neighbor_chunk_metadata,
                    neighbor_chunk_predictions,
                )
            )

            neighbor_chunk_spatialized_predictions["h3_index"] = _multiprocess_get_h3(
                neighbor_chunk_spatialized_predictions,
                _get_h3_index_for_cells,
            )

            neighbor_chunk_spatialized_predictions = (
                neighbor_chunk_spatialized_predictions[
                    neighbor_chunk_spatialized_predictions["h3_index"].isin(
                        spatialized_predictions["h3_index"]
                    )
                ]
            )

            spatialized_predictions = pd.concat(
                [spatialized_predictions, neighbor_chunk_spatialized_predictions]
            )

    spatialized_predictions = spatialized_predictions.groupby(
        ["h3_index"]
    ).prediction.agg("max")
    return spatialized_predictions
