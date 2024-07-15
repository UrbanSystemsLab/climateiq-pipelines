import collections
import csv
import itertools
import re
import os

from cloudevents import http
from google.cloud import firestore, storage
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
# ID for the Study Areas collection in Firestore.
STUDY_AREAS_COLLECTION_ID = "study_areas"
# ID for the Model collection in Firestore.
MODEL_COLLECTION_ID = "Model"
# ID for the Runs sub-collection in Firestore.
RUNS_COLLECTION_ID = "runs"


@functions_framework.cloud_event
def merge_scenario_predictions(cloud_event: http.CloudEvent):
    """Merges predictions for each chunk across scenarios into single files per chunk.

    Triggered by writes to the input bucket. If the input bucket finally contains all
    the chunks and scenarios (which are listed in Firestore), then the merge is
    performed.

    Some errors are printed instead of raised because they are non-recoverable, like
    missing files (raising errors will result in the cloud function retrying).

    Args:
        cloud_event: The CloudEvent representing the storage event.
    """
    data = cloud_event.data
    object_name = data["name"]
    match = re.match(CHUNK_FILE_NAME_PATTERN, object_name)
    # Ignore files that don't match the pattern.
    if not match:
        return

    batch_id, prediction_type, model_id, study_area_name = (
        match.group("batch_id"),
        match.group("prediction_type"),
        match.group("model_id"),
        match.group("study_area_name"),
    )

    try:
        scenario_ids = _get_expected_scenario_ids(batch_id, model_id)
        num_chunks = _get_expected_num_chunks(study_area_name)
    except ValueError as error:
        print(error)
        return

    storage_client = storage.Client()
    input_bucket = storage_client.bucket(INPUT_BUCKET_NAME)
    blobs = storage_client.list_blobs(
        INPUT_BUCKET_NAME, f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}"
    )
    chunk_ids_by_scenario_id = _get_chunk_ids_to_scenario_id(blobs)

    # If the files are not all available yet in the input bucket, don't do anything.
    try:
        files_complete = _files_complete(
            scenario_ids, num_chunks, chunk_ids_by_scenario_id
        )
    except ValueError as error:
        print(error)
        return
    if not files_complete:
        print(
            "Not all files ready for "
            f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}"
        )
        return

    # List of all unique chunk_ids. We expect the chunk_ids to be identical across
    # scenarios. Sort list alphabetically for testing purposes.
    chunk_ids = sorted(
        list(
            set(
                itertools.chain.from_iterable(
                    chunk_ids for _, chunk_ids in chunk_ids_by_scenario_id.items()
                )
            )
        ),
        key=str,
    )
    if len(chunk_ids) != num_chunks:
        print("Chunk IDs should be the same across all scenarios.")
        return

    output_bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
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
                    print(f"Not found: {error}")
                    return
                for row in rows:
                    predictions_by_h3_index[row["h3_index"]][scenario_id] = row[
                        "prediction"
                    ]
            for h3_index, predictions in predictions_by_h3_index.items():
                missing_scenario_ids = set(scenario_ids) - set(predictions.keys())
                if missing_scenario_ids:
                    print(
                        f"Not found: Missing predictions for {h3_index} for "
                        f"{', '.join(missing_scenario_ids)}."
                    )
                    return
                predictions["h3_index"] = h3_index
                # Output CSV will have the headers: h3_index,scenario_0,scenario_1...
                writer.writerow(predictions)


def _get_expected_scenario_ids(batch_id: str, model_id: str) -> list[str]:
    """Retrieves expected list of scenario_ids from run and model metadata."""
    db = firestore.Client()
    run_doc = (
        db.collection(MODEL_COLLECTION_ID)
        .document(model_id)
        .collection(RUNS_COLLECTION_ID)
        .document(batch_id)
        .get()
    )
    if not run_doc.exists:
        raise ValueError(
            f"Metadata for run {batch_id} model {model_id} does not exist "
            "in Firestore"
        )
    # Sort for predictability when testing.
    return sorted(list(run_doc.get("scenario_ids")), key=str)


def _get_expected_num_chunks(study_area_name: str) -> int:
    """Retrieves number of expected chunks per scenario from study_area metadata."""
    db = firestore.Client()
    study_area_doc = (
        db.collection(STUDY_AREAS_COLLECTION_ID).document(study_area_name).get()
    )
    if not study_area_doc.exists:
        raise ValueError(
            f"Metadata for study_area {study_area_name} does not exist in Firestore."
        )
    return study_area_doc.get("chunk_x_count") * study_area_doc.get("chunk_y_count")


def _get_chunk_ids_to_scenario_id(
    blobs: storage.Blob,
) -> dict[str, list[str]]:
    """Returns a dict of scenario_ids to list of corresponding chunk_ids."""
    chunks_per_scenario = collections.defaultdict(list)
    for blob in blobs:
        match = re.match(CHUNK_FILE_NAME_PATTERN, blob.name)
        # Ignore blobs that don't match the pattern.
        if not match:
            continue
        chunks_per_scenario[match.group("scenario_id")].append(match.group("chunk_id"))
    return chunks_per_scenario


def _files_complete(
    expected_scenario_ids: collections.abc.Iterable[str],
    expected_num_chunks: int,
    chunk_ids_by_scenario_id: collections.abc.Mapping[str, collections.abc.Sized],
) -> bool:
    """Checks if all the predictions for a batch have been written to GCS.

    * Scenarios completed match scenarios expected.
    * Each scenario has the expected number of chunks.

    Args:
        expected_scenario_ids: Iterable of scenario_ids which should be available.
        expected_num_chunks: Number of chunks per scenario.
        chunks_by_scenario_id: A Mapping of scenario_id to Iterable of chunk_ids derived
            from the filenames of existing files in GCS.
    Returns:
        True if the files are all available in GCS.
    Raises:
        ValueError: If there are more scenario_ids or chunk_ids than expected found in
            the GCS blobs.
    """
    actual_scenario_ids = set(chunk_ids_by_scenario_id.keys())
    expected_scenario_ids = set(expected_scenario_ids)
    if actual_scenario_ids == expected_scenario_ids and all(
        len(chunk_ids) == expected_num_chunks
        for _, chunk_ids in chunk_ids_by_scenario_id.items()
    ):
        return True
    elif actual_scenario_ids > expected_scenario_ids:
        raise ValueError("There are more scenario_ids in GCS than expected.")
    elif any(
        len(chunk_ids) > expected_num_chunks
        for _, chunk_ids in chunk_ids_by_scenario_id.items()
    ):
        raise ValueError(
            "There are more chunks in GCS than expected for one or more scenarios."
        )
    else:
        return False


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
