import collections
import contextlib
import io
import tempfile
from unittest import mock

from google.cloud import firestore, storage
import pytest
from typing import Any

import main


@contextlib.contextmanager
def assert_prints(message: str):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        yield
    escaped_message = message.replace("'", "'").replace('"', '\\"').replace("\n", "\\n")
    assert escaped_message in output.getvalue()


def _create_mock_doc_snap(
    items: collections.abc.Mapping[str, Any] | None = None, exists: bool = True
):
    return mock.MagicMock(exists=exists, get=(items or {}).get)


def _create_firestore_entries(
    mock_firestore_client: mock.MagicMock, scenario_ids: list[str], num_chunks: int
):
    mock_locks = mock.create_autospec(firestore.CollectionReference)
    mock_locks.document().get.return_value = mock.MagicMock(exists=False)

    mock_models = mock.create_autospec(firestore.CollectionReference)
    mock_models.document().collection().document().get.return_value = (
        _create_mock_doc_snap({"scenario_ids": scenario_ids})
    )

    mock_study_areas = mock.create_autospec(firestore.CollectionReference)
    mock_study_areas.document().get.return_value = _create_mock_doc_snap(
        {
            "chunk_x_count": num_chunks,
            "chunk_y_count": 1,
        }
    )

    def _get_collection_by_name(name: str):
        match name:
            case main.LOCKS_COLLECTION_ID:
                return mock_locks
            case main.MODEL_COLLECTION_ID:
                return mock_models
            case main.STUDY_AREAS_COLLECTION_ID:
                return mock_study_areas

    mock_firestore_client().collection.side_effect = _get_collection_by_name

    return (mock_locks, mock_models, mock_study_areas)


def _create_chunk_file(
    h3_indices_to_predictions: dict[str, float], tmp_path: str
) -> str:
    rows = ["h3_index,prediction"] + [
        f"{cell_code},{prediction}"
        for cell_code, prediction in h3_indices_to_predictions.items()
    ]
    with tempfile.NamedTemporaryFile("w+", dir=tmp_path, delete=False) as fd:
        fd.write("\n".join(rows))
    return fd.name


def _create_mock_blob(name: str, tmp_file_path: str | None = None) -> mock.MagicMock:
    blob = mock.create_autospec(storage.Blob, instance=True)
    blob.name = name
    if tmp_file_path:
        blob.open.side_effect = (
            lambda mode="r+", timeout=None, content_type=None, retry=None: open(
                tmp_file_path, mode=mode
            )
        )
        blob.exists.return_value = True
    else:
        blob.exists.return_value = False
    return blob


def _create_mock_bucket(
    tmp_files: collections.abc.Mapping[str, str | None]
) -> mock.MagicMock:
    blobs = {
        name: _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in tmp_files.items()
    }
    bucket = mock.create_autospec(storage.Bucket, instance=True)
    bucket.blob.side_effect = lambda name: blobs.get(
        name, _create_mock_blob(name, tmp_file_path=None)
    )
    return bucket


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions(
    mock_firestore_client, mock_storage_client, tmp_path
):
    mock_locks_collection, _, _ = _create_firestore_entries(
        mock_firestore_client, ["scenario0", "scenario1"], 2
    )

    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        # Chunk doesn't match file name pattern
        "batch/flood/model/nyc/scenario0/ignore/chunk0.csv": _create_chunk_file(
            {"h300": 0.99, "h301": 9.99}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk1.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
    }
    input_bucket = _create_mock_bucket(input_files)

    output_files = {
        "batch/flood/model/nyc/chunk0.csv": tmp_path / "merged_chunk0.csv",
        "batch/flood/model/nyc/chunk1.csv": tmp_path / "merged_chunk1.csv",
    }
    output_bucket = _create_mock_bucket(output_files)

    mock_storage_client().bucket.side_effect = lambda bucket_name: (
        input_bucket if bucket_name == main.INPUT_BUCKET_NAME else output_bucket
    )
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")

    assert mock_locks_collection.document().create.called
    assert mock_locks_collection.document().delete.called

    expected_chunk0_contents = (
        "cell_code,scenario0,scenario1\n" "h300,0.0,1.0\n" "h301,0.01,1.01\n"
    )
    expected_chunk1_contents = (
        "cell_code,scenario0,scenario1\n" "h310,0.1,1.1\n" "h311,0.11,1.11\n"
    )
    with open(output_files["batch/flood/model/nyc/chunk0.csv"]) as fd:
        assert fd.read() == expected_chunk0_contents
    with open(output_files["batch/flood/model/nyc/chunk1.csv"]) as fd:
        assert fd.read() == expected_chunk1_contents


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_files_incomplete_missing_scenario(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
    }
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    with assert_prints("Not all files ready"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_files_incomplete_missing_chunk(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
    }
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    with assert_prints("Not all files ready"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_run_metadata_prints_error(
    mock_firestore_client,
):
    (
        mock_firestore_client()
        .collection()
        .document()
        .collection()
        .document()
        .get.return_value
    ) = _create_mock_doc_snap(exists=False)

    with assert_prints("Metadata for run"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_study_area_metadata_prints_error(
    mock_firestore_client,
):
    mock_firestore_client().collection().document().get.return_value = (
        _create_mock_doc_snap(exists=False)
    )

    with assert_prints("Metadata for study_area"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_too_many_scenarios_prints_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk2.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
        # Extra scenario
        "batch/flood/model/nyc/scenario2/chunk0.csv": _create_chunk_file(
            {"h300": 2.00, "h301": 2.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario2/chunk1.csv": _create_chunk_file(
            {"h310": 2.10, "h311": 2.11}, tmp_path
        ),
    }
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    with assert_prints("more scenario_ids"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_too_many_chunks_prints_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk1.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
        # Extra chunk
        "batch/flood/model/nyc/scenario1/chunk2.csv": _create_chunk_file(
            {"h310": 1.20, "h311": 1.21}, tmp_path
        ),
    }
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    with assert_prints("more chunks"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_inconsistent_chunk_ids_prints_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        # Inconsistent chunk ID
        "batch/flood/model/nyc/scenario1/chunk2.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
    }
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]

    with assert_prints("Chunk IDs should be the same"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_chunk_prints_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        # Chunk is missing, but will appear in bucket listing, to simulate if a chunk
        # gets deleted in between the _files_complete check and reading the files.
        "batch/flood/model/nyc/scenario1/chunk1.csv": None,
    }
    input_bucket = _create_mock_bucket(input_files)

    output_files = {
        "batch/flood/model/nyc/chunk0.csv": tmp_path / "merged_chunk0.csv",
        "batch/flood/model/nyc/chunk1.csv": tmp_path / "merged_chunk1.csv",
    }
    output_bucket = _create_mock_bucket(output_files)

    mock_storage_client().bucket.side_effect = lambda bucket_name: (
        input_bucket if bucket_name == main.INPUT_BUCKET_NAME else output_bucket
    )
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]
    expected_error = (
        "Missing predictions for batch/flood/model/nyc/scenario1/chunk1.csv"
    )
    with assert_prints(expected_error):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_scenarios_for_cell_code_prints_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        # Has extra h3 index
        "batch/flood/model/nyc/scenario1/chunk1.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11, "h312": 1.12}, tmp_path
        ),
    }
    input_bucket = _create_mock_bucket(input_files)

    output_files = {
        "batch/flood/model/nyc/chunk0.csv": tmp_path / "merged_chunk0.csv",
    }
    output_bucket = _create_mock_bucket(output_files)

    mock_storage_client().bucket.side_effect = lambda bucket_name: (
        input_bucket if bucket_name == main.INPUT_BUCKET_NAME else output_bucket
    )
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]
    expected_error = "Not found: Missing predictions for h312 for scenario0."
    with assert_prints(expected_error):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_non_value_error_raises_error(
    mock_firestore_client, mock_storage_client, tmp_path
):
    mock_locks_collection, _, _ = _create_firestore_entries(
        mock_firestore_client, ["scenario0", "scenario1"], 2
    )
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk1.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
    }
    input_bucket = _create_mock_bucket(input_files)

    output_bucket = _create_mock_bucket({})
    output_bucket.blob.side_effect = Exception("should raise not print")

    mock_storage_client().bucket.side_effect = lambda bucket_name: (
        input_bucket if bucket_name == main.INPUT_BUCKET_NAME else output_bucket
    )
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]
    with pytest.raises(Exception, match="should raise not print"):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")

    assert mock_locks_collection.document().create.called
    assert mock_locks_collection.document().delete.called


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_lock_exists_doesnt_proceed(
    mock_firestore_client, mock_storage_client, tmp_path
):
    mock_locks, _, _ = _create_firestore_entries(
        mock_firestore_client, ["scenario0", "scenario1"], 2
    )
    mock_locks.document().get.return_value = mock.MagicMock(exists=True)

    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk0.csv": _create_chunk_file(
            {"h300": 1.00, "h301": 1.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario1/chunk1.csv": _create_chunk_file(
            {"h310": 1.10, "h311": 1.11}, tmp_path
        ),
    }
    input_bucket = _create_mock_bucket(input_files)

    output_bucket = _create_mock_bucket({})

    mock_storage_client().bucket.side_effect = lambda bucket_name: (
        input_bucket if bucket_name == main.INPUT_BUCKET_NAME else output_bucket
    )
    mock_storage_client().list_blobs.side_effect = lambda _, prefix: [
        _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in input_files.items()
        if name.startswith(prefix)
    ]
    expected_error = "already running. Terminating"
    with assert_prints(expected_error):
        main.merge_scenario_predictions("batch/flood/model/nyc/scenario0/chunk0.csv")
