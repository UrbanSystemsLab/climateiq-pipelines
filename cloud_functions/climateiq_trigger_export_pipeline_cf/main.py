from concurrent import futures
import pathlib
import json
import os
import sys
import time

from google.cloud import tasks_v2
from google.cloud.storage import client as gcs_client, retry

INPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-predictions"
OUTPUT_BUCKET_NAME = os.environ.get("BUCKET_PREFIX", "") + "climateiq-chunk-predictions"
CLIMATEIQ_PROJECT_ID = "climateiq-" + os.environ.get("BUCKET_PREFIX", "").rstrip("-")
CLIMATEIQ_EXPORT_PIPELINE_TOPIC_ID = "climateiq-spatialize-and-export-predictions"
REGION = "us-central1"
SPATIALIZE_CF_URL = (
    f"https://{REGION}-{CLIMATEIQ_PROJECT_ID}.cloudfunctions.net"
    "/spatialize-chunk-predictions"
)
SPATIALIZE_CF_SERVICE_ACCOUNT_EMAIL = (
    f"gcf-spatialize-predictions-sa@{CLIMATEIQ_PROJECT_ID}.iam.gserviceaccount.com"
)
SPATIALIZE_CF_QUEUE = "spatialize-chunk-predictions-queue"


def _write_structured_log(message: str, severity: str = "INFO"):
    print(json.dumps(dict(message=message, severity=severity)), flush=True)


def _write_file(line: str, output_filename: str, storage_client: gcs_client.Client):
    output_blob = storage_client.bucket(OUTPUT_BUCKET_NAME).blob(output_filename)
    # Specify retry here due to bug:
    # https://github.com/googleapis/python-storage/issues/1242
    output_blob.upload_from_string(line, retry=retry.DEFAULT_RETRY)


def trigger_export_pipeline(object_name: str) -> None:
    """Triggered by writes to the "climateiq-predictions" bucket.

    Splits predictions into one file per chunk and kicks off
    climateiq_spatialize_chunk_predictions cloud function for each chunk.

    Note: This function only runs once all output prediction files are written.
    Additionally, the climateiq_spatialize_chunk_predictions cloud function is
    only triggered once all prediction files per chunk are written since data
    from neighboring chunks is required for spatialization.

    Args:
        object_name: The name of the blob written to the bucket.
    """
    start = time.time()

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
    storage_client = gcs_client.Client()
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

    _write_structured_log(f"[{object_name}] Starting process.", "DEBUG")

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

    futures.wait(write_futures, return_when=futures.FIRST_EXCEPTION)
    # If any exceptions were raised in a thread, calling result() will raise it here.
    for future in write_futures:
        future.result()

    _write_structured_log(
        (
            f"[{object_name}] Created {len(output_filenames)} files in "
            f"{time.time() - start} s."
        ),
        "DEBUG",
    )

    # Once all output files have been written, push tasks to Task Queue.
    tasks_client = tasks_v2.CloudTasksClient()
    queue_futures = []
    with futures.ThreadPoolExecutor() as executor:
        for output_filename in output_filenames:
            task = tasks_v2.Task(
                http_request=tasks_v2.HttpRequest(
                    http_method=tasks_v2.HttpMethod.POST,
                    url=SPATIALIZE_CF_URL,
                    headers={"Content-type": "application/json"},
                    oidc_token=tasks_v2.OidcToken(
                        service_account_email=SPATIALIZE_CF_SERVICE_ACCOUNT_EMAIL,
                        audience=SPATIALIZE_CF_URL,
                    ),
                    body=json.dumps({"object_name": output_filename}).encode("utf-8"),
                ),
            )
            queue_futures.append(
                executor.submit(
                    tasks_client.create_task,
                    tasks_v2.CreateTaskRequest(
                        parent=tasks_client.queue_path(
                            CLIMATEIQ_PROJECT_ID,
                            REGION,
                            SPATIALIZE_CF_QUEUE,
                        ),
                        task=task,
                    ),
                )
            )
    futures.wait(queue_futures, return_when=futures.FIRST_EXCEPTION)
    # If any exceptions were raised in a thread, calling result() will raise it here.
    for future in write_futures:
        future.result()

    _write_structured_log(
        (
            f"[{object_name}] Triggered export pipeline for "
            f"{len(output_filenames)} chunks."
        ),
        "DEBUG",
    )


if __name__ == "__main__":
    trigger_export_pipeline(sys.argv[1])
