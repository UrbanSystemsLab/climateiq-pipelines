import functions_framework
import pathlib

from cloudevents.http import CloudEvent
from google.cloud import firestore
from typing import Optional

STUDY_AREAS = "study_areas"


# Triggered by the "object finalized" Cloud Storage event type.
@functions_framework.cloud_event
def export_model_predictions(cloud_event: CloudEvent) -> Optional[dict]:
    """This function is triggered when a new object is created or an existing
    object is overwritten in the "climateiq-predictions" storage bucket.

    Args:
        cloud_event: The CloudEvent representing the storage event.
    Returns:
        A dictionary containing the study area metadata or None if the study
        area is not found.
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

    return _get_study_area_metadata(study_area_name)


def _get_study_area_metadata(study_area_name: str) -> Optional[dict]:
    """Retrieves metadata for a given study area from Firestore.

    Args:
        study_area_name: The name of the study area to retrieve metadata for.
    Returns:
        A dictionary containing the study area metadata or None if the study
        area is not found.
    """
    db = firestore.Client()

    # Retrieve study area metadata from Firestore
    study_area_ref = db.collection(STUDY_AREAS).document(study_area_name)
    study_area = study_area_ref.get()

    if not study_area.exists:
        raise ValueError(f'Study area "{study_area_name}" does not exist')

    return study_area.to_dict()
