from io import StringIO

from cloudevents import http
from google.cloud import firestore
from google.cloud import storage
import pandas as pd
from pandas import testing as pd_testing
import pytest
import main
from unittest import mock


def test_export_model_predictions_invalid_object_name() -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "invalid_name",  # Invalid object name
    }
    event = http.CloudEvent(attributes, data)

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert "Invalid object name format. Expected 5 components." in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_missing_study_area(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    mock_firestore_client().collection("").document(
        ""
    ).get().exists = False  # Indicate study area doesn't exist

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Study area "study-area-name" does not exist' in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_study_area(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }  # Missing "cell_size" required field
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Study area "study-area-name" is missing one or more required '
        "fields: cell_size, crs, chunks" in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_missing_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "missing-chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }  # Missing "row_count" required field
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Chunk "chunk-id" is missing one or more required '
        "fields: row_count, col_count, x_ll_corner, y_ll_corner" in str(exc_info.value)
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_missing_predictions(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = ""
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert "Predictions file is missing predictions." in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_too_many_predictions(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = (
        '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
        '{"instance": [2], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    )
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert "Predictions file has too many predictions" in str(exc_info.value)


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_h3_centroids_within_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3], [4, 5, 6]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "row_count": 2,
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build expected output data
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
        main.export_model_predictions(event)

    pd_testing.assert_series_equal(
        pd.read_json(StringIO(str(exc_info.value)), typ="series"),
        expected_series,
        check_dtype=False,
    )


@mock.patch.object(storage, "Client", autospec=True)
@mock.patch.object(firestore, "Client", autospec=True)
def test_export_model_predictions_h3_centroids_outside_chunk(
    mock_firestore_client, mock_storage_client
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/chunk-id",
    }
    event = http.CloudEvent(attributes, data)

    # Build mock Storage object
    predictions = '{"instance": [1], "prediction": [[1, 2, 3, 4, 5, 6], \
    [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], \
    [19, 20, 21, 22, 23, 24]]}\n'
    with mock_storage_client().bucket("").blob("").open() as mock_fd:
        mock_fd.__iter__.return_value = iter(predictions.splitlines())

    # Build mock Firestore document
    metadata = {
        "name": "study_area_name",
        "cell_size": 5,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "row_count": 4,
                "col_count": 6,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },
    }
    mock_firestore_client().collection().document().get().to_dict.return_value = (
        metadata
    )

    # Build expected output data
    expected_series = pd.Series(
        {
            "8d8f2c80c1582bf": 6.0,
            "8d8f2c80c15863f": 3.0,
            "8d8f2c80c15867f": 7.5,  # Average of prediction values 4, 5, 10, 11
            "8d8f2c80c1586bf": 2.0,
            "8d8f2c80c1586ff": 9.0,
            "8d8f2c80c15b83f": 23.5,  # Average of prediction values 23, 24
            "8d8f2c80c15b93f": 15.0,  # Average of prediction values 12, 18
            "8d8f2c80c15b9bf": 16.5,  # Average of prediction values 16, 17
            "8d8f2c80c15bc3f": 19.5,  # Average of prediction values 19, 20
            "8d8f2c80c15bd3f": 11.0,  # Average of prediction values 8, 14
            "8d8f2c80c15bd7f": 18.0,  # Average of prediction values 15, 21
        }
    )

    with pytest.raises(NotImplementedError) as exc_info:
        main.export_model_predictions(event)

    pd_testing.assert_series_equal(
        pd.read_json(StringIO(str(exc_info.value)), typ="series"),
        expected_series,
        check_dtype=False,
    )
