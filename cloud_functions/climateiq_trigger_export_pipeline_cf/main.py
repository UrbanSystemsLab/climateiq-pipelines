import functions_framework
import pathlib
import json
import os

from google import cloud
from cloudevents import http

INPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-predictions"
OUTPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-chunk-predictions"
CLIMATEIQ_PROJECT_ID = "climateiq"
CLIMATEIQ_EXPORT_PIPELINE_TOPIC_ID = "climateiq-spatialize-and-export-predictions"


@functions_framework.cloud_event
def trigger_export_pipeline(cloud_event: http.CloudEvent) -> None:
    """Triggered by writes to the "climateiq-predictions" bucket.

    Splits predictions into one file per chunk and kicks off
    climateiq_spatialize_chunk_predictions cloud function for each chunk.

    Note: This function only runs once all output prediction files are written.
    Additionally, the climateiq_spatialize_chunk_predictions cloud function is
    only triggered once all prediction files per chunk are written since data
    from neighboring chunks is required for spatializiation.

    Args:
        cloud_event: The CloudEvent representing the storage event.

    Raises:
        ValueError: If the object name format is invalid.
    """
    data = cloud_event.data
    object_name = data["name"]

    # Extract components from the object name and determine the total number of
    # output prediction files.
    path = pathlib.PurePosixPath(object_name)
    if len(path.parts) != 6:
        raise ValueError(
            "Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/prediction.results-"
            "<file_number>-of-{number_of_files_generated}'"
        )
    id, prediction_type, model_id, study_area_name, scenario_id, filename = path.parts
    if filename.count("-") != 3:
        raise ValueError(
            "Invalid object name format. Expected format: '<id>/<prediction_type>/"
            "<model_id>/<study_area_name>/<scenario_id>/prediction.results-"
            "<file_number>-of-{number_of_files_generated}'"
        )
    _, _, _, file_count = filename.split("-")
    total_prediction_files = int(file_count)

    # Retrieve all input prediction files.
    storage_client = cloud.storage.Client()
    input_blobs = storage_client.list_blobs(
        INPUT_BUCKET_NAME,
        prefix=f"{id}/{prediction_type}/{model_id}/{study_area_name}/{scenario_id}",
    )
    total_input_blobs = len(input_blobs)
    if total_input_blobs != total_prediction_files:
        # Return early since all expected output prediction files have not been
        # written yet.
        return

    # Split predictions into one file per chunk and output to GCS.
    output_filenames = []
    for blob in input_blobs:
        with blob.open() as fd:
            for line in fd:
                chunk_id = json.loads(line)["instance"]["key"]
                output_filename = (
                    f"{id}/{prediction_type}/{model_id}/"
                    f"{study_area_name}/{scenario_id}/{chunk_id}"
                )
                output_filenames.append(output_filename)
                output_blob = storage_client.bucket(OUTPUT_BUCKET_NAME).blob(
                    output_filename
                )
                output_blob.upload_from_string(line)

    # Once all output files have been written, publish pubsub message per chunk to kick
    # off export pipeline.
    publisher = cloud.pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(
        CLIMATEIQ_PROJECT_ID, CLIMATEIQ_EXPORT_PIPELINE_TOPIC_ID
    )
    for output_filename in output_filenames:
        future = publisher.publish(
            topic_path,
            data=output_filename.encode("utf-8"),
            origin="climateiq_trigger_export_pipeline_cf",
        )
        future.result()
