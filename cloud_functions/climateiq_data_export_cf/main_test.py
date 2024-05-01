import main
import pytest

from cloudevents.http import CloudEvent
from unittest import mock


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_object_name(
    mock_firestore_client,
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "invalid_name",  # Invalid object name
    }
    event = CloudEvent(attributes, data)

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert "Invalid object name format. Expected 5 components." in str(
        exc_info.value
    )
    assert not mock_firestore_client.called


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_study_area(
    mock_firestore_client,
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id",
    }
    event = CloudEvent(attributes, data)

    mock_study_area = mock.Mock()
    mock_study_area.exists = False  # Indicate study area doesn't exist
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_study_area
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Study area "study-area-name" does not exist' in str(exc_info.value)


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_no_chunks(mock_firestore_client) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id",
    }
    event = CloudEvent(attributes, data)

    mock_study_area = mock.Mock()
    mock_study_area.exists = True
    expected_metadata = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_study_area
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_missing_chunk_id(
    mock_firestore_client,
) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id",
    }
    event = CloudEvent(attributes, data)

    mock_study_area = mock.Mock()
    mock_study_area.exists = True
    expected_metadata = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "chunks": {"missing-chunk-id": {"col_count": 10, "row_count": 5}},
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_study_area
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions(mock_firestore_client) -> None:
    attributes = {
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "source",
    }
    data = {
        "bucket": "climateiq-predictions",
        "name": "prediction-type/model-id/study-area-name/scenario-id/"
        "chunk-id",
    }
    event = CloudEvent(attributes, data)

    mock_study_area = mock.Mock()
    mock_study_area.exists = True
    expected_metadata = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "chunks": {"chunk-id": {"col_count": 10, "row_count": 5}},
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_study_area
        )

    chunk_metadata = main.export_model_predictions(event)
    assert chunk_metadata == expected_metadata["chunks"]["chunk-id"]
