import collections
import csv
import re
import os

import flask
from google.cloud import storage
import functions_framework


# Bucket name, where spatialized prediction outputs are stored.
INPUT_BUCKET_NAME = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-spatialized-chunk-predictions"
)
# Bucket name, where merged prediction outputs are stored.
OUTPUT_BUCKET_NAME = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-spatialized-merged-predictions"
)
# File name pattern for the CSVs for each scenario and chunk.
CHUNK_FILE_NAME_PATTERN = (
    r"(?P<batch_id>\w+)/(?P<prediction_type>\w+)/(?P<model_id>\w+)/"
    r"(?P<study_area_name>\w+)/(?P<scenario_id>\w+)/(?P<chunk_id>\w+)\.csv"
)


@functions_framework.http
def merge_scenario_predictions(request: flask.Request) -> tuple[str, int]:
    """Merges predictions for each chunk across scenarios into single files per chunk.

    TODO: Trigger based on file writes instead.

    Args:
        request: A Flask request with the query parameters: batch_id,
            prediction_type, model_id, study_area_name.
    Returns:
        A tuple of the HTTP response (message, status_code).
    """
    try:
        batch_id, prediction_type, model_id, study_area_name = _get_args(
            request, ("batch_id", "prediction_type", "model_id", "study_area_name")
        )
    except ValueError as error:
        return f"Bad request: {error}", 400

    storage_client = storage.Client()
    input_bucket = storage_client.bucket(INPUT_BUCKET_NAME)
    output_bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)

    blobs = storage_client.list_blobs(
        INPUT_BUCKET_NAME, f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}"
    )
    chunk_ids, scenario_ids = _get_chunk_and_scenario_ids(blobs)
    for chunk_id in chunk_ids:
        output_file_name = (
            f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}/{chunk_id}.csv"
        )
        blob_to_write = output_bucket.blob(output_file_name)
        with blob_to_write.open("w") as fd:
            # Open the blob and start writing a CSV file with the headers
            # h3_index,scenario_0,scenario_1...
            writer = csv.DictWriter(fd, fieldnames=["h3_index"] + scenario_ids)
            writer.writeheader()
            predictions_by_h3_index: dict[str, dict] = collections.defaultdict(dict)
            for scenario_id in scenario_ids:
                object_name = (
                    f"{batch_id}/{prediction_type}/{model_id}/"
                    f"{study_area_name}/{scenario_id}/{chunk_id}.csv"
                )
                try:
                    rows = _get_file_content(input_bucket, object_name)
                except ValueError as error:
                    return f"Not found: {error}", 404
                for row in rows:
                    predictions_by_h3_index[row["h3_index"]][scenario_id] = row[
                        "prediction"
                    ]
            for h3_index, predictions in predictions_by_h3_index.items():
                missing_scenario_ids = set(scenario_ids) - set(predictions.keys())
                if missing_scenario_ids:
                    return (
                        (
                            f"Not found: Missing predictions for {h3_index} for "
                            f"{', '.join(missing_scenario_ids)}."
                        ),
                        404,
                    )
                predictions["h3_index"] = h3_index
                # Output CSV will have the headers: h3_index,scenario_0,scenario_1...
                writer.writerow(predictions)
    return "Success", 200


def _get_args(
    request: flask.Request, arg_names: collections.abc.Iterable[str]
) -> list[str]:
    """Gets the args from the Flask request.

    Args:
        request: The request to get query args from.
        arg_names: An Iterable of the names of the args to get.
    Returns:
        A list of the arg values ordered by arg_names.
    Raises:
        ValueError: If the query arg is not found in the request.
    """
    args = []
    for arg_name in arg_names:
        arg_value = request.args.get(arg_name)
        if not arg_value:
            raise ValueError(f"Missing arg {arg_name}")
        args.append(arg_value)
    return args


def _get_chunk_and_scenario_ids(
    blobs: list[storage.Blob],
) -> tuple[list[str], list[str]]:
    """Gets the chunk_ids and scenario_ids from a list of Blobs.

    We assume that every chunk_id and scenario_id combination is valid.

    This ignores files which don't match the spatialized output file pattern.

    Args:
        blobs: List of Blobs to look through.
    Returns:
        A tuple of (chunk_ids, scenario_ids).
    """
    chunk_ids = set()
    scenario_ids = set()
    for blob in blobs:
        match = re.match(CHUNK_FILE_NAME_PATTERN, blob.name)
        # Ignore blobs that don't match the pattern.
        if not match:
            continue
        chunk_ids.add(match.group("chunk_id"))
        scenario_ids.add(match.group("scenario_id"))
    return sorted(list(chunk_ids), key=str), sorted(list(scenario_ids), key=str)


def _get_file_content(bucket: storage.Bucket, object_name: str) -> list[dict]:
    """Gets the content from a Blob.

    Assumes Blob content is in CSV format with headers h3_index,prediction...

    Args:
        bucket: The GCS bucket the Blob is in.
        object_name: The name of the Blob.
    Returns:
        Blob contents as a list of dict rows.
    Raises:
        ValueError: If the Blob doesn't exist.
    """
    blob = bucket.blob(object_name)
    # If the specific blob doesn't exist (i.e., no predictions for given scenario_id and
    # chunk_id), then raise an error.
    if not blob.exists():
        raise ValueError(f"Missing predictions for {object_name}")
    with blob.open() as fd:
        return list(csv.DictReader(fd))
