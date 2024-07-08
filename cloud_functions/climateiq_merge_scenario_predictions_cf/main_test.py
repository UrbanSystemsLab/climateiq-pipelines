import re
import tempfile

import flask
from google.cloud import storage
import pytest
from unittest import mock

import main


# Create a fake "app" for generating test request contexts.
@pytest.fixture(scope="module")
def app():
    return flask.Flask(__name__)


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


def _create_mock_blob(
    name: str, tmp_file_path: str | None = None, exists: bool = True
) -> mock.MagicMock:
    blob = mock.create_autospec(storage.Blob, instance=True)
    blob.name = name
    if tmp_file_path:
        blob.open.side_effect = lambda mode="r+": open(tmp_file_path, mode=mode)
    blob.exists.return_value = exists
    return blob


def _create_mock_bucket(tmp_files: dict[str, str]) -> mock.MagicMock:
    blobs = {
        name: _create_mock_blob(name, tmp_file_path)
        for name, tmp_file_path in tmp_files.items()
    }
    bucket = mock.create_autospec(storage.Bucket, instance=True)
    bucket.blob.side_effect = lambda name: blobs.get(
        name, _create_mock_blob(name, tmp_file_path=None, exists=False)
    )
    return bucket


@mock.patch.object(storage, "Client", autospec=True)
def test_merge_scenario_predictions(mock_storage_client, tmp_path, app) -> None:
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
    with app.test_request_context(
        query_string={
            "batch_id": "batch",
            "prediction_type": "flood",
            "model_id": "model",
            "study_area_name": "nyc",
        }
    ):
        result = main.merge_scenario_predictions(flask.request)
        assert result == ("Success", 200)

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
def test_merge_scenario_predictions_missing_chunk_raises_error(
    mock_storage_client, tmp_path, app
) -> None:
    input_files = {
        "batch/flood/model/nyc/scenario0/chunk0.csv": _create_chunk_file(
            {"h300": 0.00, "h301": 0.01}, tmp_path
        ),
        "batch/flood/model/nyc/scenario0/chunk1.csv": _create_chunk_file(
            {"h310": 0.10, "h311": 0.11}, tmp_path
        ),
        # Chunk only exists in one scenario
        "batch/flood/model/nyc/scenario0/chunk2.csv": _create_chunk_file(
            {"h320": 0.20, "h321": 0.21}, tmp_path
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
        "batch/flood/model/nyc/chunk2.csv": tmp_path / "merged_chunk2.csv",
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
    with app.test_request_context(
        query_string={
            "batch_id": "batch",
            "prediction_type": "flood",
            "model_id": "model",
            "study_area_name": "nyc",
        }
    ):
        result = main.merge_scenario_predictions(flask.request)
        assert result == (
            (
                "Not found: Missing predictions for "
                "batch/flood/model/nyc/scenario1/chunk2.csv"
            ),
            404,
        )


@mock.patch.object(storage, "Client", autospec=True)
def test_merge_scenario_predictions_missing_scenario_for_chunk_raises_error(
    mock_storage_client, tmp_path, app
) -> None:
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
        # Scenario only exists for one chunk
        "batch/flood/model/nyc/scenario2/chunk0.csv": _create_chunk_file(
            {"h300": 2.00, "h301": 2.01}, tmp_path
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
    with app.test_request_context(
        query_string={
            "batch_id": "batch",
            "prediction_type": "flood",
            "model_id": "model",
            "study_area_name": "nyc",
        }
    ):
        result = main.merge_scenario_predictions(flask.request)
        assert result == (
            (
                "Not found: Missing predictions for "
                "batch/flood/model/nyc/scenario2/chunk1.csv"
            ),
            404,
        )


@mock.patch.object(storage, "Client", autospec=True)
def test_merge_scenario_predictions_missing_scenarios_for_h3_index_raises_error(
    mock_storage_client, tmp_path, app
) -> None:
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
    with app.test_request_context(
        query_string={
            "batch_id": "batch",
            "prediction_type": "flood",
            "model_id": "model",
            "study_area_name": "nyc",
        }
    ):
        result = main.merge_scenario_predictions(flask.request)
        assert result == (
            ("Not found: Missing predictions for h312 for " "scenario0."),
            404,
        )


@pytest.mark.parametrize(
    "arg_name", ["batch_id", "prediction_type", "model_id", "study_area_name"]
)
def test_merge_scenario_predictions_missing_args(arg_name, app):
    query_string_args = {
        "batch_id": "batch",
        "prediction_type": "flood",
        "model_id": "model",
        "study_area_name": "nyc",
    }
    del query_string_args[arg_name]
    with app.test_request_context(query_string=query_string_args):
        result_msg, result_code = main.merge_scenario_predictions(flask.request)
        assert re.match(f"Bad request:.*{arg_name}.*", result_msg)
        assert result_code == 400
