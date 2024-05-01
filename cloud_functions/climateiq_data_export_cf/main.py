import functions_framework
import pathlib

from cloudevents.http import CloudEvent
from google.cloud import firestore

STUDY_AREAS_ID = "study_areas"


# Triggered by the "object finalized" Cloud Storage event type.
@functions_framework.cloud_event
def export_model_predictions(cloud_event: CloudEvent) -> dict:
    """This function is triggered when a new object is created or an existing
    object is overwritten in the "climateiq-predictions" storage bucket.

    Args:
        cloud_event: The CloudEvent representing the storage event.
    Returns:
        A dictionary containing metadata for the chunk.
    Raises:
        ValueError: If the object name format is invalid, the study area does
        not exist or the chunk does not exist.
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

    return chunk_metadata


def _get_study_area_metadata(study_area_name: str) -> dict:
    """Retrieves metadata for a given study area from Firestore.

    Args:
        study_area_name: The name of the study area to retrieve metadata for.
    Returns:
        A dictionary containing metadata for the study area.
    Raises:
        ValueError: If the study area does not exist.
    """
    db = firestore.Client()

    study_area_ref = db.collection(STUDY_AREAS_ID).document(study_area_name)
    study_area_doc = study_area_ref.get()

    if not study_area_doc.exists:
        raise ValueError(f'Study area "{study_area_name}" does not exist')

    return study_area_doc.to_dict()


def _get_chunk_metadata(study_area_metadata: dict, chunk_id: str) -> dict:
    """Retrieves metadata for a specific chunk within a study area.

    Args:
        study_area_metadata (dict): A dictionary containing metadata for the
        study area.
        chunk_id (str): The unique identifier of the chunk to retrieve
        metadata for.
    Returns:
        A dictionary containing metadata for the chunk
    Raises:
        ValueError: If the specified chunk does not exist.
    """
    chunks = study_area_metadata.get("chunks", {})
    chunk_metadata = chunks.get(chunk_id)

    if chunk_metadata is None:
        raise ValueError(f'Chunk "{chunk_id}" does not exist')

    return chunk_metadata
