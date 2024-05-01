import functions_framework
import pathlib
import geopandas as gpd
import pandas as pd
import numpy as np

from cloudevents.http import CloudEvent
from google.cloud import firestore

STUDY_AREAS_ID = "study_areas"
GLOBAL_CRS = "EPSG:4326"


# Triggered by the "object finalized" Cloud Storage event type.
@functions_framework.cloud_event
def export_model_predictions(cloud_event: CloudEvent) -> dict:
    """This function is triggered when a new object is created or an existing
    object is overwritten in the "climateiq-predictions" storage bucket.

    Args:
        cloud_event: The CloudEvent representing the storage event.
    Returns:
        pd.DataFrame: A DataFrame with latitude and longitude coordinates for each cell's center in the raster chunk.
    Raises:
        ValueError: If the object name format, study area metadata or chunk area metadata is invalid.
    """
    data = cloud_event.data
    name = data["name"]

    # Extract components from the object name
    path = pathlib.PurePosixPath(name)
    if len(path.parts) != 5:
        raise ValueError("Invalid object name format. Expected 5 components.")

    prediction_type, model_id, study_area_name, scenario_id, chunk_id = (
        path.parts
    )

    try:
        study_area_metadata = _get_study_area_metadata(study_area_name)
    except ValueError as ve:
        raise ve

    try:
        chunk_metadata = _get_chunk_metadata(study_area_metadata, chunk_id)
    except ValueError as ve:
        raise ve

    return _build_spatialized_model_predictions(
        study_area_metadata, chunk_metadata
    )


def _get_study_area_metadata(study_area_name: str) -> dict:
    """Retrieves metadata for a given study area from Firestore.

    Args:
        study_area_name: The name of the study area to retrieve metadata for.
    Returns:
        A dictionary containing metadata for the study area.
    Raises:
        ValueError: If the study area does not exist or its metadata is missing required
        fields.
    """
    db = firestore.Client()

    study_area_ref = db.collection(STUDY_AREAS_ID).document(study_area_name)
    study_area_doc = study_area_ref.get()

    if not study_area_doc.exists:
        raise ValueError(f'Study area "{study_area_name}" does not exist')

    study_area_metadata = study_area_doc.to_dict()
    if (
        study_area_metadata.get("cell_size") is None
        or study_area_metadata.get("crs") is None
        or study_area_metadata.get("chunks") is None
    ):
        raise ValueError(
            f'Study area "{study_area_name}" is missing one or more required field(s): cell_size, crs, chunks'
        )

    return study_area_metadata


def _get_chunk_metadata(study_area_metadata: dict, chunk_id: str) -> dict:
    """Retrieves metadata for a specific chunk within a study area.

    Args:
        study_area_metadata (dict): A dictionary containing metadata for the
        study area.
        chunk_id (str): The unique identifier of the chunk to retrieve
        metadata for.
    Returns:
        A dictionary containing metadata for the chunk.
    Raises:
        ValueError: If the specified chunk does not exist or its metadata is missing required fields.
    """
    chunks = study_area_metadata["chunks"]
    chunk_metadata = chunks.get(chunk_id)

    if chunk_metadata is None:
        raise ValueError(f'Chunk "{chunk_id}" does not exist')

    if (
        chunk_metadata.get("row_count") is None
        or chunk_metadata.get("col_count") is None
        or chunk_metadata.get("x_ll_corner") is None
        or chunk_metadata.get("y_ll_corner") is None
    ):
        raise ValueError(
            f'Chunk "{chunk_id}" is missing one or more required fields: row_count, col_count, x_ll_corner, y_ll_corner'
        )

    return chunk_metadata


# TODO: Parse bucket contents and append predictions.
def _build_spatialized_model_predictions(
    study_area_metadata: dict, chunk_metadata: dict
):
    """Builds a DataFrame with latitude and longitude coordinates for each cell's center in the raster chunk.

    Args:
        study_area_metadata: A dictionary containing metadata for the study area.
        chunk_metadata: A dictionary containing metadata for the chunk.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'lat' and 'lon'.
    """
    rows = np.arange(chunk_metadata["row_count"])
    cols = np.arange(chunk_metadata["col_count"])

    # Calculate cell center coordinates in the raster's CRS.
    x_centers = (
        chunk_metadata["x_ll_corner"]
        + (cols + 0.5) * study_area_metadata["cell_size"]
    )
    y_centers = (
        chunk_metadata["y_ll_corner"]
        + (rows + 0.5) * study_area_metadata["cell_size"]
    )

    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    # Convert 2D meshgrids to 1D arrays.
    x_coords = x_grid.flatten()
    y_coords = y_grid.flatten()

    # Convert coordinates from the raster's CRS to the global CRS (lat/lon).
    gdf_src_crs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=study_area_metadata["crs"],
    )
    gdf_global_crs = gdf_src_crs.to_crs(GLOBAL_CRS)
    return pd.DataFrame(
        {"lat": gdf_global_crs.geometry.y, "lon": gdf_global_crs.geometry.x}
    )
