import base64
import json
import pandas as pd
import pathlib
import os
import numpy as np
import functions_framework
import geopandas as gpd

from typing import Any
from cloudevents import http
from google.cloud import firestore, storage
from shapely import geometry
from h3 import h3

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


# Triggered from a message on the "climateiq-spatialize-and-export-predictions"
# Pub/Sub topic.
@functions_framework.cloud_event
def subscribe(cloud_event: http.CloudEvent) -> None:
    """This function spatializes model predictions for a single chunk and outputs a
    CSV file to GCS containing H3 indexes along with associated predictions.

    Args:
        cloud_event: The CloudEvent representing the Pub/Sub message.

    Raises:
        ValueError: If the object name format, study area metadata, chunk / neighbor
        chunk metadata or predictions file format is invalid.
    """
    object_name = base64.b64decode(cloud_event.data["message"]["data"]).decode()

    # Extract components from the object name.
    path = pathlib.PurePosixPath(object_name)
    if len(path.parts) != 6:
        raise ValueError(
            "Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'"
        )

    id, prediction_type, model_id, study_area_name, scenario_id, chunk_id = path.parts

    predictions = _read_chunk_predictions(object_name)
    study_area_metadata, chunks_ref = _get_study_area_metadata(study_area_name)
    chunk_metadata = _get_chunk_metadata(study_area_metadata, chunk_id)

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

    storage_client = storage.Client()
    bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
    blob = bucket.blob(
        f"{id}/{prediction_type}/{model_id}/{study_area_name}/{scenario_id}/{chunk_id}"
        ".csv"
    )
    with blob.open("w+") as fd:
        h3_predictions.to_csv(fd)


# TODO: Modify this logic once CNN output schema is confirmed. Also update to
# account for errors and special values.
def _read_chunk_predictions(object_name: str) -> np.ndarray:
    """Reads model predictions for a given chunk from GCS and outputs
    these predictions in a 2D array.

    Args:
        object_name: The name of the chunk object to read.

    Returns:
        A 2D array containing the model predictions for the chunk.

    Raises:
        ValueError: If the predictions file format is invalid.
    """
    storage_client = storage.Client()
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
    """Reads model predictions for a neighbor chunk from GCS and outputs
    these predictions in a 2D array.

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
            "Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>"
        )
    *prefix, current_chunk_id = path.parts
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
    db = firestore.Client()

    study_area_ref = db.collection(STUDY_AREAS_ID).document(study_area_name)
    chunks_ref = study_area_ref.collection(CHUNKS_ID)
    study_area_doc = study_area_ref.get()

    if not study_area_doc.exists:
        raise ValueError(f'Study area "{study_area_name}" does not exist')

    study_area_metadata = study_area_doc.to_dict()
    if (
        "cell_size" not in study_area_metadata
        or "crs" not in study_area_metadata
        or "chunks" not in study_area_metadata
        or "row_count" not in study_area_metadata
        or "col_count" not in study_area_metadata
    ):
        raise ValueError(
            f'Study area "{study_area_name}" is missing one or more required '
            "fields: cell_size, crs, chunks, row_count, col_count"
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


def _get_chunk_metadata(study_area_metadata: dict, chunk_id: str) -> dict:
    """Retrieves metadata for a specific chunk within a study area.

    Args:
        study_area_metadata: A dictionary containing metadata for the
        study area.
        chunk_id: The id of the chunk to retrieve metadata for.

    Returns:
        A dictionary containing metadata for the chunk.

    Raises:
        ValueError: If the specified chunk does not exist or its metadata is
        missing required fields.
    """
    chunks = study_area_metadata["chunks"]
    chunk_metadata = chunks.get(chunk_id)

    if chunk_metadata is None:
        raise ValueError(f'Chunk "{chunk_id}" does not exist')

    if not _chunk_metadata_fields_valid(chunk_metadata):
        raise ValueError(
            f'Chunk "{chunk_id}" is missing one or more required fields: '
            "row_count, col_count, x_ll_corner, y_ll_corner, x_index, y_index"
        )

    return chunk_metadata


def _build_spatialized_model_predictions(
    study_area_metadata: dict, chunk_metadata: dict, predictions: np.ndarray
) -> pd.DataFrame:
    """Builds a DataFrame containing the lat/lon coordinates of each cell's
    center point.

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


def _add_h3_index_details(cell: pd.Series) -> pd.Series:
    """Projects the cell centroid to a H3 index.

    Args:
        cell: A cell row containing the lat and lon of the cell centroid.

    Returns:
        A Series containing H3 information for the projected cell centroid.
    """
    h3_index = h3.geo_to_h3(cell["lat"], cell["lon"], H3_LEVEL)
    centroid_lat, centroid_lon = h3.h3_to_geo(h3_index)
    boundary_xy = h3.h3_to_geo_boundary(h3_index, True)
    return pd.Series(
        {
            "h3_index": h3_index,
            "h3_centroid_lat": centroid_lat,
            "h3_centroid_lon": centroid_lon,
            "h3_boundary": geometry.Polygon(boundary_xy),
        }
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
    chunk and de-dupes duplicate H3 projections by using the average prediction
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
    spatialized_predictions[
        ["h3_index", "h3_centroid_lat", "h3_centroid_lon", "h3_boundary"]
    ] = spatialized_predictions.apply(_add_h3_index_details, axis=1)

    # Filter out any rows where the projected H3 centroid falls outside of the
    # chunk boundary.
    chunk_boundary = _get_chunk_boundary(study_area_metadata, chunk_metadata)
    spatialized_predictions = spatialized_predictions[
        spatialized_predictions.apply(
            lambda row: chunk_boundary.contains(
                geometry.Point(row["h3_centroid_lon"], row["h3_centroid_lat"])
            ),
            axis=1,
        )
    ]

    # Extract rows where the projected H3 cell is not fully contained within the chunk
    # so we can aggregate prediction values across chunk boundaries.
    boundary_h3_cells = spatialized_predictions[
        spatialized_predictions.apply(
            lambda row: not row["h3_boundary"].within(chunk_boundary),
            axis=1,
        )
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
        ).prediction.agg("mean")
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
            or neighbor_x >= study_area_metadata["col_count"]
            or neighbor_y >= study_area_metadata["row_count"]
        ):
            # Chunk is outside the study area boundary.
            continue
        query = (
            chunks_ref.where("x_index", "==", neighbor_x)
            .where("y_index", "==", neighbor_y)
            .limit(1)
        )
        chunk_doc = query.get()
        if not chunk_doc.exists:
            raise ValueError(
                f"Neighbor chunk at index {neighbor_x, neighbor_y} is missing from the "
                "study area."
            )

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
            # TODO: Optionally only calculate the h3_index if calculating other
            # metadata is expensive
            neighbor_chunk_spatialized_predictions[
                ["h3_index", "h3_centroid_lat", "h3_centroid_lon", "h3_boundary"]
            ] = neighbor_chunk_spatialized_predictions.apply(
                _add_h3_index_details, axis=1
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
    ).prediction.agg("mean")
    return spatialized_predictions
