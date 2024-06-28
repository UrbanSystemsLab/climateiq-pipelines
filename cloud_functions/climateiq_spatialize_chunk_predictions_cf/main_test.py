import base64
import main
import pytest
import pandas as pd

from cloudevents import http
from io import StringIO
from pandas import testing as pd_testing
from unittest import mock
from typing import Any, Dict
from google.cloud import storage, firestore


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

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        "Invalid object name format. Expected format: '<id>/<prediction_type>/"
        "<model_id>/<study_area_name>/<scenario_id>/<chunk_id>'" in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    mock_firestore_client().collection("").document(
        ""
    ).get().exists = False  # Indicate study area doesn't exist

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert 'Study area "study-area-name" does not exist' in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 0,
                "y_index": 0,
            }
        },
    }  # Missing "cell_size" required field
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        'Study area "study-area-name" is missing one or more required '
        "fields: cell_size, crs, chunks" in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "missing-chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 0,
                "y_index": 0,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 0,
                "y_index": 0,
            }
        },
    }  # Missing "row_count" required field
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        'Chunk "chunk-id" is missing one or more required '
        "fields: row_count, col_count, x_ll_corner, y_ll_corner" in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 0,
                "y_index": 0,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        "Predictions file: id/prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id is missing." in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 0,
                "y_index": 0,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert "Predictions file has too many predictions" in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build neighbor chunk data.
    (
        mock_firestore_client().collection().document().collection()
    ).where().where().limit().get().exists = False  # Neighbor chunks do not exist.

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert "Neighbor chunk at index (0, 1) is missing from the study area" in str(
        exc_info.value
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 3,
                "col_count": 2,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build neighbor chunk data.
    neighbor_metadata = metadata["chunks"]["chunk-id"].copy()
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().id = "neighbor-chunk-id"
    neighbor_metadata.pop("row_count")  # Missing "row_count" required field
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().to_dict.return_value = neighbor_metadata

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        "Neighbor chunk at index (0, 1) is missing one or more required fields: id,"
        " row_count, col_count, x_ll_corner,y_ll_corner, x_index, y_index"
        in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
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
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 3,
                "col_count": 2,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build neighbor chunk data.
    neighbor_metadata = metadata["chunks"]["chunk-id"].copy()
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().id = "neighbor-chunk-id"
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().to_dict.return_value = neighbor_metadata

    with pytest.raises(ValueError) as exc_info:
        main.subscribe(event)

    assert (
        "Predictions file: id/prediction-type/model-id/study-area-name/scenario-id/"
        "neighbor-chunk-id is missing."
    ) in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_spatialize_chunk_predictions_h3_centroids_within_chunk(
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
        mock_fd.__iter__.side_effect = predictions.splitlines().__iter__

    # Build mock Firestore document
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build neighbor chunk data.
    neighbor_metadata = metadata["chunks"]["chunk-id"].copy()
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().id = "neighbor-chunk-id"
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().to_dict.return_value = neighbor_metadata

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
        }
    )

    with pytest.raises(NotImplementedError) as exc_info:
        main.subscribe(event)

    pd_testing.assert_series_equal(
        pd.read_json(StringIO(str(exc_info.value)), typ="series"),
        expected_series,
        check_dtype=False,
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_spatialize_chunk_predictions_h3_centroids_outside_chunk(
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
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[1, 2, 3, 4, 5, 6],'
        "[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18],"
        "[19, 20, 21, 22, 23, 24]]}"
    )

    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.side_effect = predictions.splitlines().__iter__

    # Build mock Firestore document
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 5,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 4,
                "col_count": 6,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build neighbor chunk data.
    neighbor_metadata = metadata["chunks"]["chunk-id"].copy()
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().id = "neighbor-chunk-id"
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().to_dict.return_value = neighbor_metadata

    # Build expected output data (neighbor chunks have same data as current chunk in
    # this test so prediction values stay the same after aggregation.)
    expected_series = pd.Series(
        {
            "8d8f2c80c1582bf": 6.0,
            "8d8f2c80c15863f": 3.0,
            "8d8f2c80c15867f": 7.5,
            "8d8f2c80c1586bf": 2.0,
            "8d8f2c80c1586ff": 9.0,
            "8d8f2c80c15b83f": 23.5,  # Average of prediction values
            # 23, 24 (from current chunk)
            "8d8f2c80c15b93f": 15.0,  # Average of prediction values
            # 12, 18 (from current chunk)
            "8d8f2c80c15b9bf": 16.5,  # Average of prediction values
            # 16, 17 (from current chunk)
            "8d8f2c80c15bc3f": 19.5,  # Average of prediction values
            # 19, 20 (from current chunk)
            "8d8f2c80c15bd3f": 11.0,  # Average of prediction values
            # 8, 14 (from current chunk)
            "8d8f2c80c15bd7f": 18.0,  # Average of prediction values
            # 15, 21 (from current chunk)
        }
    )

    with pytest.raises(NotImplementedError) as exc_info:
        main.subscribe(event)

    pd_testing.assert_series_equal(
        pd.read_json(StringIO(str(exc_info.value)), typ="series"),
        expected_series,
        check_dtype=False,
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_spatialize_chunk_predictions_overlapping_neighbors(
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
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1}, '
        '"prediction": [[1, 2, 3, 4, 5, 6],'
        "[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],"
        "[25, 26, 27, 28, 29, 30]]}"
    )
    predictions_bottom = (
        '{"instance":  {"values": [1, 2, 3, 4], "key": 1},'
        '"prediction": [[31, 32, 33, 34, 35, 36],'
        "[37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48], [49, 50, 51, 52, 53, 54],"
        "[55, 56, 57, 58, 59, 60]]}"
    )

    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.side_effect = [
            iter(predictions.splitlines()),
            iter(predictions_bottom.splitlines()),
        ]  # Current chunk + 1 intersecting neighbor

    # Build mock Firestore document
    metadata: Dict[str, Any] = {
        "name": "study_area_name",
        "cell_size": 3,
        "crs": "EPSG:32618",
        "row_count": 2,
        "col_count": 3,
        "chunks": {
            "chunk-id": {
                "row_count": 5,
                "col_count": 6,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
                "x_index": 1,
                "y_index": 1,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

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
    ).limit().get().id.side_effect = [
        "neighbor-chunk-left",
        "neighbor-chunk-right",
        "neighbor-chunk-bottom-left",
        "neighbor-chunk-bottom-right",
        "neighbor-chunk-bottom",
    ]
    (
        mock_firestore_client().collection().document().collection().where().where()
    ).limit().get().to_dict.side_effect = [
        neighbor_left,
        neighbor_right,
        neighbor_bottom_left,
        neighbor_bottom_right,
        neighbor_bottom,
    ]

    # Build expected output data
    expected_series = pd.Series(
        {
            "8d8f2c80c1586ff": 8.0,  # Average of prediction values 10, 11, 12,
            # 4, 5, 6 (from current chunk)
            "8d8f2c80c15bc3f": 26.2857142857,  # Average of prediction values 25, 26,
            # 27, 20, 21, (from current chunk) and 32, 33 (from bottom neighbor chunk)
            "8d8f2c80c15bd3f": 11.5,  # Average of prediction values  14, 15, 8,
            # 9 (from current chunk)
            "8d8f2c80c15bd7f": 22.714285714,  # Average of prediction values 28, 29,
            # 22, 23, 24, 16, 17 (from current chunk)
        }
    )

    with pytest.raises(NotImplementedError) as exc_info:
        main.subscribe(event)

    pd_testing.assert_series_equal(
        pd.read_json(StringIO(str(exc_info.value)), typ="series"),
        expected_series,
        check_dtype=False,
    )
