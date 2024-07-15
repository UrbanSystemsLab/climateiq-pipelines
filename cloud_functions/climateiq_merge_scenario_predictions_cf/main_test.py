import collections
import contextlib
import io
import tempfile
import typing
from unittest import mock

from cloudevents import http
from google.cloud import firestore, storage

import main


def _create_pubsub_event() -> http.CloudEvent:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "batch/flood/model/nyc/scenario0/chunk0.csv",
    }
    return http.CloudEvent(attributes, data)


def _create_mock_doc_snap(
    items: collections.abc.Mapping[str, typing.Any] | None = None, exists: bool = True
):
    return mock.MagicMock(exists=exists, get=(items or {}).get)


def _create_firestore_entries(
    mock_firestore_client: mock.MagicMock, scenario_ids: list[str], num_chunks: int
):
    (
        mock_firestore_client()
        .collection()
        .document()
        .collection()
        .document()
        .get.return_value
    ) = _create_mock_doc_snap({"scenario_ids": scenario_ids})

    mock_firestore_client().collection().document().get.return_value = (
        _create_mock_doc_snap(
            {
                "chunk_x_count": num_chunks,
                "chunk_y_count": 1,
            }
        )
    )


def _create_chunk_file(
    h3_indices_to_predictions: dict[str, float], tmp_path: str
) -> str:
    rows = ["h3_index,prediction"] + [
        f"{h3_index},{prediction}"
        for h3_index, prediction in h3_indices_to_predictions.items()
    ]
    with tempfile.NamedTemporaryFile("w+", dir=tmp_path, delete=False) as fd:
        fd.write("\n".join(rows))
    return fd.name


def _create_mock_blob(name: str, tmp_file_path: str | None = None) -> mock.MagicMock:
    blob = mock.create_autospec(storage.Blob, instance=True)
    blob.name = name
    if tmp_file_path:
        blob.open.side_effect = lambda mode="r+": open(tmp_file_path, mode=mode)
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
    _create_firestore_entries(mock_firestore_client, ["scenario0", "scenario1"], 2)

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

    main.merge_scenario_predictions(_create_pubsub_event())

    expected_chunk0_contents = (
        "h3_index,scenario0,scenario1\n" "h300,0.0,1.0\n" "h301,0.01,1.01\n"
    )
    expected_chunk1_contents = (
        "h3_index,scenario0,scenario1\n" "h310,0.1,1.1\n" "h311,0.11,1.11\n"
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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "Not all files ready" in output.getvalue()


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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "Not all files ready" in output.getvalue()


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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "Metadata for run" in output.getvalue()


@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_study_area_metadata_prints_error(
    mock_firestore_client,
):
    mock_firestore_client().collection().document().get.return_value = (
        _create_mock_doc_snap(exists=False)
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "Metadata for study_area" in output.getvalue()


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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "more scenario_ids" in output.getvalue()


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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "more chunks" in output.getvalue()


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

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert "Chunk IDs should be the same" in output.getvalue()


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
        "Not found: Missing predictions for batch/flood/model/nyc/scenario1/chunk1.csv"
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert expected_error in output.getvalue()


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_merge_scenario_predictions_missing_scenarios_for_h3_index_prints_error(
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
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.merge_scenario_predictions(_create_pubsub_event())
    assert expected_error in output.getvalue()
