from concurrent import futures
import functions_framework
import pathlib
import json
import os
import time

from google.cloud import pubsub_v1, storage
from cloudevents import http

INPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-predictions"
OUTPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-chunk-predictions"
CLIMATEIQ_PROJECT_ID = "climateiq-" + os.environ.get("BUCKET_PREFIX", "").rstrip("-")
CLIMATEIQ_EXPORT_PIPELINE_TOPIC_ID = "climateiq-spatialize-and-export-predictions"


def _write_file(line: str, output_filename: str, storage_client: storage.Client):
    output_blob = storage_client.bucket(OUTPUT_BUCKET_NAME).blob(output_filename)
    # Specify retry here due to bug:
    # https://github.com/googleapis/python-storage/issues/1242
    output_blob.upload_from_string(line, retry=storage.retry.DEFAULT_RETRY)


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
    """
    start = time.time()

    data = cloud_event.data
    object_name = data["name"]

    # Structured logging:
    # https://cloud.google.com/functions/docs/monitoring/logging#writing_structured_logs
    print(json.dumps(dict(severity="DEBUG", message=f"[{object_name}] CF started")))

    # Extract components from the object name and determine the total number of
    # output prediction files.
    expected_format = (
        "<id>/<prediction_type>/<model_id>/<study_area_name>/"
        "<scenario_id>/prediction.results-<file_number>-of-{number_of_files_generated}"
    )
    path = pathlib.PurePosixPath(object_name)
    if len(path.parts) != 6:
        raise ValueError(
            f"Invalid object name format. Expected format: '{expected_format}'\n"
            f"Actual name: '{object_name}'"
        )
    id, prediction_type, model_id, study_area_name, scenario_id, filename = path.parts
    if filename.count("-") != 3:
        raise ValueError(
            f"Invalid object name format. Expected format: '{expected_format}'\n"
            f"Actual name: '{object_name}'"
        )
    _, _, _, file_count = filename.split("-")
    if prediction_type != "flood":
        raise ValueError(
            "Export pipeline can currently only be used for flood predictions."
        )
    try:
        total_prediction_files = int(file_count)
    except ValueError:
        raise ValueError(
            f"Invalid object name format. Expected format: '{expected_format}'\n"
            f"Actual name: '{object_name}'"
        )

    # Retrieve all input prediction files.
    storage_client = storage.Client()
    input_blobs = list(
        storage_client.list_blobs(
            INPUT_BUCKET_NAME,
            prefix=(
                f"{id}/{prediction_type}/{model_id}/{study_area_name}/"
                f"{scenario_id}/prediction.results"
            ),
        )
    )
    total_input_blobs = len(input_blobs)
    if total_input_blobs != total_prediction_files:
        # Return early since all expected output prediction files have not been
        # written yet.
        return

    # Split predictions into one file per chunk and output to GCS.
    output_filenames = []
    write_futures = []
    with futures.ThreadPoolExecutor() as executor:
        for blob in input_blobs:
            with blob.open() as fd:
                for line in fd:
                    chunk_id = json.loads(line)["instance"]["key"]
                    output_filename = (
                        f"{id}/{prediction_type}/{model_id}/"
                        f"{study_area_name}/{scenario_id}/{chunk_id}"
                    )
                    output_filenames.append(output_filename)
                    future = executor.submit(
                        _write_file, line, output_filename, storage_client
                    )
                    write_futures.append(future)

    futures.wait(write_futures)
    print(
        json.dumps(
            dict(
                severity="DEBUG",
                message=(
                    f"[{object_name}] Created {len(output_filenames)} files in "
                    f"{time.time() - start} s."
                ),
            )
        )
    )

    # Once all output files have been written, publish pubsub message per chunk to kick
    # off export pipeline.
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(
        CLIMATEIQ_PROJECT_ID, CLIMATEIQ_EXPORT_PIPELINE_TOPIC_ID
    )
    publish_futures = []
    for output_filename in output_filenames:
        future = publisher.publish(
            topic_path,
            data=output_filename.encode("utf-8"),
            origin="climateiq_trigger_export_pipeline_cf",
        )
        publish_futures.append(future)
    futures.wait(publish_futures)
    print(
        json.dumps(
            dict(
                severity="DEBUG",
                message=(
                    f"[{object_name}] Started export pipeline for "
                    f"{len(output_filenames)} chunks."
                ),
            )
        )
    )
