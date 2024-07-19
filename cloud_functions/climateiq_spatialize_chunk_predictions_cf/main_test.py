import base64
import tempfile
import io
import contextlib
import pandas as pd

from cloudevents import http
from google.cloud import firestore_v1
from google.cloud import storage
from typing import Any, Dict, List
from unittest import mock

import main


def _create_tmpfile(contents: str, dir: str) -> str:
    with tempfile.NamedTemporaryFile("w", dir=dir, delete=False) as fd:
        fd.write(contents)
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
        blob_name: _create_mock_blob(blob_name, tmp_file_path)
        for blob_name, tmp_file_path in tmp_files.items()
    }
    bucket = mock.create_autospec(storage.Bucket, instance=True)
    bucket.blob.side_effect = lambda name: (
        blobs.get(name, _create_mock_blob(name, tmp_file_path=None, exists=False))
    )
    return bucket


def test_spatialize_chunk_predictions_invalid_object_name() -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(b"invalid_name"),
            }
        },
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        "Invalid object name format. Expected format: '<id>/<prediction_type>/"
        "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'"
        "\nActual name: 'invalid_name'" in output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_missing_study_area(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().exists = False  # Indicate study area doesn't exist

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert 'Study area "study-area-name" does not exist' in output.getvalue()


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_invalid_study_area(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }  # Missing "cell_size" required field
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 0,
            "y_index": 0,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        'Study area "study-area-name" is missing one or more required '
        "fields: cell_size, crs, row_count, col_count" in output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_missing_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "missing-chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 0,
            "y_index": 0,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().exists = False

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in output.getvalue()


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_invalid_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 0,
            "y_index": 0,  # Missing "row_count" required field
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        'Chunk "chunk-id" is missing one or more required '
        "fields: row_count, col_count, x_ll_corner, y_ll_corner" in output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_missing_predictions(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = ""
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 0,
            "y_index": 0,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        "Predictions file: id/prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id is missing." in output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_too_many_predictions(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}\n'
        '{"instance": {"values": [1, 2, 3, 4], "key": 2},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = predictions.splitlines()

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 0,
            "y_index": 0,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunk_ref = mock_study_area_ref.collection("chunks").document("chunk-id")
    mock_chunk_ref.get().to_dict.return_value = chunks_metadata[0]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert "Predictions file has too many predictions" in output.getvalue()


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_missing_expected_neighbor_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]
    mock_chunks_ref.where().where().limit().get.return_value = (
        []
    )  # Neighbor chunks do not exist.

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)
    assert "Neighbor chunk at index (0, 1) is missing from the study area" in str(
        output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_invalid_neighbor_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = [predictions]

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 3,
            "col_count": 2,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    neighbor_metadata_mock = mock.MagicMock()
    neighbor_metadata_mock.id = "neighbor-chunk-id"
    neighbor_metadata = chunks_metadata[0].copy()
    neighbor_metadata.pop("row_count")  # Missing "row_count" required field
    neighbor_metadata_mock.to_dict.return_value = neighbor_metadata
    mock_chunks_ref.where().where().limit().get.return_value = [neighbor_metadata_mock]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        "Neighbor chunk at index (0, 1) is missing one or more required fields: id,"
        " row_count, col_count, x_ll_corner,y_ll_corner, x_index, y_index"
        in output.getvalue()
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_neighbor_chunk_missing_predictions(
    mock_firestore_client, mock_storage_client
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(
            predictions.splitlines()
        )  # Predictions for current chunk only

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 3,
            "col_count": 2,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    neighbor_metadata_mock = mock.MagicMock()
    neighbor_metadata_mock.id = "neighbor-chunk-id"
    neighbor_metadata = chunks_metadata[0].copy()
    neighbor_metadata_mock.to_dict.return_value = neighbor_metadata
    mock_chunks_ref.where().where().limit().get.return_value = [neighbor_metadata_mock]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main.spatialize_chunk_predictions(event)

    assert (
        "Predictions file: id/prediction-type/model-id/study-area-name/scenario-id/"
        "neighbor-chunk-id is missing."
    ) in output.getvalue()


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_h3_centroids_within_chunk(
    mock_firestore_client, mock_storage_client, tmp_path
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance": {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3], [4, 5, 6]]}'
    )
    output_file_path = tmp_path / "output.csv"
    mock_bucket = _create_mock_bucket(
        {
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id": (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-id"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id.csv": (
                output_file_path
            ),
        }
    )
    mock_storage_client().bucket.return_value = mock_bucket

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 2,
            "col_count": 3,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    neighbor_metadata_mock = mock.MagicMock()
    neighbor_metadata_mock.id = "neighbor-chunk-id"
    neighbor_metadata = chunks_metadata[0].copy()
    neighbor_metadata_mock.to_dict.return_value = neighbor_metadata
    mock_chunks_ref.where().where().limit().get.return_value = [neighbor_metadata_mock]

    # Build expected output data (neighbor chunks have same data as current chunk in
    # this test so prediction values stay the same after aggregation.)
    expected_series = pd.Series(
        {
            "8d8f2c80c1582bf": 3.0,
            "8d8f2c80c1586bf": 1.0,
            "8d8f2c80c1586ff": 2.0,
            "8d8f2c80c15b83f": 6.0,
            "8d8f2c80c15bc3f": 4.0,
            "8d8f2c80c15bd7f": 5.0,
        },
        name="prediction",
    )
    expected_series.index.name = "h3_index"

    main.spatialize_chunk_predictions(event)

    pd.testing.assert_series_equal(
        pd.read_csv(output_file_path, index_col=0)["prediction"],
        expected_series,
        check_dtype=False,
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_h3_centroids_outside_chunk(
    mock_firestore_client, mock_storage_client, tmp_path
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3, 4, 5, 6],'
        "[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18],"
        "[19, 20, 21, 22, 23, 24]]}"
    )
    output_file_path = tmp_path / "output.csv"
    mock_bucket = _create_mock_bucket(
        {
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id": (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-id"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id.csv": (
                output_file_path
            ),
        }
    )
    mock_storage_client().bucket.return_value = mock_bucket

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 5,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 4,
            "col_count": 6,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    neighbor_metadata_mock = mock.MagicMock()
    neighbor_metadata_mock.id = "neighbor-chunk-id"
    neighbor_metadata = chunks_metadata[0].copy()
    neighbor_metadata_mock.to_dict.return_value = neighbor_metadata
    mock_chunks_ref.where().where().limit().get.return_value = [neighbor_metadata_mock]

    # Build expected output data (neighbor chunks have same data as current chunk in
    # this test so prediction values stay the same after aggregation.)
    expected_series = pd.Series(
        {
            "8d8f2c80c1582bf": 6.0,
            "8d8f2c80c15863f": 3.0,
            "8d8f2c80c15867f": 11,
            "8d8f2c80c1586bf": 2.0,
            "8d8f2c80c1586ff": 9.0,
            "8d8f2c80c15b83f": 24.0,  # Max of prediction values
            # 23, 24 (from current chunk)
            "8d8f2c80c15b93f": 18.0,  # Max of prediction values
            # 12, 18 (from current chunk)
            "8d8f2c80c15b9bf": 17.0,  # Max of prediction values
            # 16, 17 (from current chunk)
            "8d8f2c80c15bc3f": 20.0,  # Max of prediction values
            # 19, 20 (from current chunk)
            "8d8f2c80c15bd3f": 14.0,  # Max of prediction values
            # 8, 14 (from current chunk)
            "8d8f2c80c15bd7f": 21.0,  # Max of prediction values
            # 15, 21 (from current chunk)
        },
        name="prediction",
    )
    expected_series.index.name = "h3_index"

    main.spatialize_chunk_predictions(event)

    pd.testing.assert_series_equal(
        pd.read_csv(output_file_path, index_col=0)["prediction"],
        expected_series,
        check_dtype=False,
    )


@mock.patch.object(storage.client, "Client", autospec=True)
@mock.patch.object(firestore_v1, "Client", autospec=True)
def test_spatialize_chunk_predictions_overlapping_neighbors(
    mock_firestore_client, mock_storage_client, tmp_path
) -> None:
    event = http.CloudEvent(
        {
            "type": "google.cloud.pubsub.topic.v1.messagePublished",
            "source": "source",
        },
        {
            "message": {
                "data": base64.b64encode(
                    b"id/prediction-type/model-id/study-area-name/scenario-id/chunk-id"
                ),
            }
        },
    )

    # Build mock Storage object
    predictions = (
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1}, '
        '"prediction": [[1, 2, 3, 4, 5, 6],'
        "[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],"
        "[25, 26, 27, 28, 29, 30]]}\n"
    )
    predictions_bottom = (
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[31, 32, 33, 34, 35, 36],'
        "[37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48], [49, 50, 51, 52, 53, 54],"
        "[55, 56, 57, 58, 59, 60]]}\n"
    )
    output_file_path = tmp_path / "output.csv"
    mock_bucket = _create_mock_bucket(
        {
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id": (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-left"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-right"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-bottom-left"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-bottom-right"
            ): (
                _create_tmpfile(
                    predictions,
                    tmp_path,
                )
            ),
            # 1 intersecting neighbor
            (
                "id/prediction-type/model-id/study-area-name/scenario-id/"
                "neighbor-chunk-bottom"
            ): (
                _create_tmpfile(
                    predictions_bottom,
                    tmp_path,
                )
            ),
            "id/prediction-type/model-id/study-area-name/scenario-id/chunk-id.csv": (
                output_file_path
            ),
        }
    )
    mock_storage_client().bucket.return_value = mock_bucket

    # Build mock Firestore document
    study_area_metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 3,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
    }
    chunks_metadata: List[Dict[str, Any]] = [
        {
            "id": "chunk-id",
            "row_count": 5,
            "col_count": 6,
            "x_ll_corner": 500,
            "y_ll_corner": 100,
            "x_index": 1,
            "y_index": 1,
        }
    ]
    mock_study_area_ref = (
        mock_firestore_client().collection("study_areas").document("study-area-name")
    )
    mock_study_area_ref.get().to_dict.return_value = study_area_metadata
    mock_chunks_ref = mock_study_area_ref.collection("chunks")
    mock_chunks_ref.get.return_value = chunks_metadata
    mock_chunks_ref.document("chunk-id").get().to_dict.return_value = chunks_metadata[0]

    # Build neighbor chunk data.
    neighbor_left = {
        "row_count": 5,
        "col_count": 6,
        "x_ll_corner": 482,
        "y_ll_corner": 100,
        "x_index": 0,
        "y_index": 1,
    }
    neighbor_right = {
        "row_count": 5,
        "col_count": 6,
        "x_ll_corner": 518,
        "y_ll_corner": 100,
        "x_index": 2,
        "y_index": 1,
    }
    neighbor_bottom_left = {
        "row_count": 5,
        "col_count": 6,
        "x_ll_corner": 482,
        "y_ll_corner": 85,
        "x_index": 0,
        "y_index": 0,
    }
    neighbor_bottom_right = {
        "row_count": 5,
        "col_count": 6,
        "x_ll_corner": 518,
        "y_ll_corner": 85,
        "x_index": 2,
        "y_index": 0,
    }
    neighbor_bottom = {
        "row_count": 5,
        "col_count": 6,
        "x_ll_corner": 500,
        "y_ll_corner": 85,
        "x_index": 1,
        "y_index": 0,
    }
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get.side_effect = [
        [
            mock.MagicMock(
                **{"id": "neighbor-chunk-left", "to_dict.return_value": neighbor_left}
            )
        ],
        [
            mock.MagicMock(
                **{"id": "neighbor-chunk-right", "to_dict.return_value": neighbor_right}
            )
        ],
        [
            mock.MagicMock(
                **{
                    "id": "neighbor-chunk-bottom-left",
                    "to_dict.return_value": neighbor_bottom_left,
                }
            )
        ],
        [
            mock.MagicMock(
                **{
                    "id": "neighbor-chunk-bottom-right",
                    "to_dict.return_value": neighbor_bottom_right,
                }
            )
        ],
        [
            mock.MagicMock(
                **{
                    "id": "neighbor-chunk-bottom",
                    "to_dict.return_value": neighbor_bottom,
                }
            )
        ],
    ]

    # Build expected output data
    expected_series = pd.Series(
        {
            "8d8f2c80c1586ff": 12.0,  # Max of prediction values 10, 11, 12,
            # 4, 5, 6 (from current chunk)
            "8d8f2c80c15bc3f": 33.0,  # Max of prediction values 25, 26,
            # 27, 20, 21, (from current chunk) and 32, 33 (from bottom neighbor chunk)
            "8d8f2c80c15bd3f": 15.0,  # Max of prediction values  14, 15, 8,
            # 9 (from current chunk)
            "8d8f2c80c15bd7f": 29,  # Max of prediction values 28, 29,
            # 22, 23, 24, 16, 17 (from current chunk)
        },
        name="prediction",
    )
    expected_series.index.name = "h3_index"

    main.spatialize_chunk_predictions(event)

    pd.testing.assert_series_equal(
        pd.read_csv(output_file_path, index_col=0)["prediction"],
        expected_series,
        check_dtype=False,
    )
