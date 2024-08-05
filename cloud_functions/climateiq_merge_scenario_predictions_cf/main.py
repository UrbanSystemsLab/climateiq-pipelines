import collections
from concurrent import futures
import csv
import itertools
import json
import re
import os
import sys
import time

from google.cloud import firestore, storage


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
    r"(?P<batch_id>[^/]+)/(?P<prediction_type>[^/]+)/(?P<model_id>[^/]+)/"
    r"(?P<study_area_name>[^/]+)/(?P<scenario_id>[^/]+)/(?P<chunk_id>[^/]+)\.csv"
)
# ID for the Study Areas collection in Firestore.
STUDY_AREAS_COLLECTION_ID = "study_areas"
# ID for the Model collection in Firestore.
MODEL_COLLECTION_ID = "models"
# ID for the Runs sub-collection in Firestore.
RUNS_COLLECTION_ID = "runs"
# ID for the Locks sub-collection in Firestore.
LOCKS_COLLECTION_ID = "locks"
# Time in seconds for writes to GCS to time out.
WRITE_TIMEOUT = 360
# Max number of workers for the process pool. If the job is running out of CPU or
# memory, lower this. Defaults to None i.e. use the default value for
# ProcessPoolExecutor.
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", 0)) or None


def _write_structured_log(message: str, severity: str = "INFO"):
    print(json.dumps(dict(message=message, severity=severity)), flush=True)


def _set_lock(batch_id: str, prediction_type: str, model_id: str, study_area_name: str):
    """Creates a Firestore entry to indicate a merge is happening."""
    db = firestore.Client()
    # Use hyphens to make calls to endpoints through URL simpler.
    db.collection(LOCKS_COLLECTION_ID).document(
        f"{batch_id}-{prediction_type}-{model_id}-{study_area_name}"
    ).create({"running": True})


def _get_lock(batch_id: str, prediction_type: str, model_id: str, study_area_name: str):
    """Checks if a Firestore entry for this merge operation already exists."""
    db = firestore.Client()
    lock_doc = (
        db.collection(LOCKS_COLLECTION_ID)
        .document(f"{batch_id}-{prediction_type}-{model_id}-{study_area_name}")
        .get()
    )
    return lock_doc.exists


def _delete_lock(
    batch_id: str,
    prediction_type: str,
    model_id: str,
    study_area_name: str,
):
    """Deletes the lock."""
    db = firestore.Client()
    (
        db.collection(LOCKS_COLLECTION_ID)
        .document(f"{batch_id}-{prediction_type}-{model_id}-{study_area_name}")
        .delete()
    )


def merge_scenario_predictions(object_name: str):
    """Merges predictions for each chunk across scenarios into single files per chunk.

    Triggered by writes to the input bucket. If the input bucket finally contains all
    the chunks and scenarios (which are listed in Firestore), then the merge is
    performed.

    Some errors are printed instead of raised because they are non-recoverable, like
    missing files (raising errors will result in the cloud function retrying).

    Args:
        object_name: The name of the object which triggered this job.
    """
    start = time.time()

    match = re.match(CHUNK_FILE_NAME_PATTERN, object_name)
    # Ignore files that don't match the pattern.
    if match is None:
        _write_structured_log(
            f"Invalid object name format. Expected format: '<id>/<prediction_type>/"
            f"<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'\n"
            f"Actual name: '{object_name}'",
            "WARNING",
        )
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
        _write_structured_log(str(error), "ERROR")
        return

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        INPUT_BUCKET_NAME,
        prefix=f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}",
    )
    chunk_ids_by_scenario_id = _get_chunk_ids_to_scenario_id(blobs)

    # If the files are not all available yet in the input bucket, don't do anything.
    try:
        files_complete = _files_complete(
            scenario_ids, num_chunks, chunk_ids_by_scenario_id
        )
    except ValueError as error:
        _write_structured_log(str(error), "ERROR")
        return
    if not files_complete:
        _write_structured_log(
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
        _write_structured_log(
            f"[{batch_id}/{prediction_type}/{model_id}/{study_area_name}] "
            "Chunk IDs should be the same across all scenarios.",
            "ERROR",
        )
        return

    _write_structured_log(
        f"[{batch_id}/{prediction_type}/{model_id}/{study_area_name}] Starting merge.",
        "DEBUG",
    )

    if _get_lock(batch_id, prediction_type, model_id, study_area_name):
        _write_structured_log(
            f"[{batch_id}/{prediction_type}/{model_id}/{study_area_name}] "
            "Merge already running. Terminating this process.",
            "WARNING",
        )
        return
    _set_lock(batch_id, prediction_type, model_id, study_area_name)

    max_chunks_per_process = max(
        int(num_chunks / max(((os.cpu_count() or 2) - 1), 1)), 1
    )
    subset_futures = []
    with futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        for i in range(0, num_chunks, max_chunks_per_process):
            future = executor.submit(
                _merge_chunk_set,
                batch_id,
                prediction_type,
                model_id,
                study_area_name,
                scenario_ids,
                chunk_ids[i : i + max_chunks_per_process],
            )
            subset_futures.append(future)
    futures.wait(subset_futures, return_when=futures.FIRST_EXCEPTION)

    for future in subset_futures:
        # Trigger raising any errors.
        try:
            future.result()
        # If it's a ValueError, then it's non-recoverable (e.g. file doesn't exist) and
        # we don't want to raise an actual error. All other error types should be
        # raised.
        except ValueError as e:
            _delete_lock(batch_id, prediction_type, model_id, study_area_name)
            _write_structured_log(str(e), "ERROR")
            return
        except:  # noqa: E722
            _delete_lock(batch_id, prediction_type, model_id, study_area_name)
            raise

    _delete_lock(batch_id, prediction_type, model_id, study_area_name)

    _write_structured_log(
        f"[{batch_id}/{prediction_type}/{model_id}/{study_area_name}] Wrote "
        f"{len(chunk_ids)} files in {time.time() - start} s.",
        "DEBUG",
    )

    return


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

    Assumes Blob content is in CSV format with headers cell_code,prediction...

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


def _merge_chunk_set(
    batch_id: str,
    prediction_type: str,
    model_id: str,
    study_area_name: str,
    scenario_ids: list[str],
    chunk_ids: list[str],
):
    """Kicks off merge operations for a set of chunks. Called in subprocess."""
    # Re-initialize the bucket here because this is called in process.
    input_bucket = storage.Client().bucket(INPUT_BUCKET_NAME)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_NAME)
    write_futures = []
    with futures.ThreadPoolExecutor() as executor:
        for chunk_id in chunk_ids:
            write_futures.append(
                executor.submit(
                    _merge_single_chunk,
                    batch_id,
                    prediction_type,
                    model_id,
                    study_area_name,
                    scenario_ids,
                    chunk_id,
                    input_bucket,
                    output_bucket,
                )
            )
    futures.wait(write_futures, return_when=futures.FIRST_EXCEPTION)
    # Trigger raising any errors.
    for future in write_futures:
        future.result()


def _merge_single_chunk(
    batch_id: str,
    prediction_type: str,
    model_id: str,
    study_area_name: str,
    scenario_ids: list[str],
    chunk_id: str,
    input_bucket: storage.Bucket,
    output_bucket: storage.Bucket,
):
    """Reads data across scenarios and writes merged file for a single chunk."""
    read_futures = []
    with futures.ThreadPoolExecutor() as executor:
        for scenario_id in scenario_ids:
            object_name = (
                f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}/"
                f"{scenario_id}/{chunk_id}.csv"
            )
            future = executor.submit(_get_file_content, input_bucket, object_name)
            read_futures.append((scenario_id, future))
    futures.wait(
        [future for _, future in read_futures], return_when=futures.FIRST_EXCEPTION
    )

    predictions_by_cell_code: dict[str, dict] = collections.defaultdict(dict)
    for scenario_id, future in read_futures:
        for row in future.result():
            predictions_by_cell_code[row["h3_index"]][scenario_id] = row["prediction"]

    output_file_name = (
        f"{batch_id}/{prediction_type}/{model_id}/{study_area_name}/{chunk_id}.csv"
    )
    blob_to_write = output_bucket.blob(output_file_name)
    with blob_to_write.open(
        "w",
        timeout=WRITE_TIMEOUT,
        content_type="text/csv",
        retry=storage.retry.DEFAULT_RETRY,
    ) as fd:
        # Open the blob and start writing a CSV file with the headers
        # cell_code,scenario_0,scenario_1...
        writer = csv.DictWriter(fd, fieldnames=["cell_code"] + scenario_ids)
        writer.writeheader()
        for cell_code, predictions in predictions_by_cell_code.items():
            missing_scenario_ids = set(scenario_ids) - set(predictions.keys())
            if missing_scenario_ids:
                raise ValueError(
                    f"Not found: Missing predictions for {cell_code} for "
                    f"{', '.join(missing_scenario_ids)}."
                )
            predictions["cell_code"] = cell_code
            # Output CSV will have the headers: cell_code,scenario_0,scenario_1...
            writer.writerow(predictions)


if __name__ == "__main__":
    merge_scenario_predictions(sys.argv[1])
