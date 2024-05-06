import main
import pytest
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from google.cloud import storage
from google.cloud import firestore
from cloudevents.http import CloudEvent
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
    event = CloudEvent(attributes, data)

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert "Invalid object name format. Expected 5 components." in str(
        exc_info.value
    )


@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_missing_study_area(
    mock_firestore_client, mock_storage_client
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

    # Build mock Storage object
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_blob = mock.create_autospec(storage.Blob)
    predictions = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    mock_blob.download_as_text.return_value = json.dumps(predictions)
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.bucket.return_value = mock_bucket

    # Build mock Firestore document
    mock_document = mock.create_autospec(firestore.DocumentSnapshot)
    mock_document.exists = False  # Indicate study area doesn't exist
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
    mock_document.to_dict.return_value = metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_document
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Study area "study-area-name" does not exist' in str(exc_info.value)


@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_study_area(
    mock_firestore_client, mock_storage_client
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

    # Build mock Storage object
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_blob = mock.create_autospec(storage.Blob)
    predictions = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    mock_blob.download_as_text.return_value = json.dumps(predictions)
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.bucket.return_value = mock_bucket

    # Build mock Firestore document
    mock_document = mock.create_autospec(firestore.DocumentSnapshot)
    mock_document.exists = True
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
    mock_document.to_dict.return_value = metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_document
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Study area "study-area-name" is missing one or more required '
        "fields: cell_size, crs, chunks" in str(exc_info.value)
    )


@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_missing_chunk(
    mock_firestore_client, mock_storage_client
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

    # Build mock Storage object
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_blob = mock.create_autospec(storage.Blob)
    predictions = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    mock_blob.download_as_text.return_value = json.dumps(predictions)
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.bucket.return_value = mock_bucket

    # Build mock Firestore document
    mock_document = mock.create_autospec(firestore.DocumentSnapshot)
    mock_document.exists = True
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
    mock_document.to_dict.return_value = metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_document
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_chunk(
    mock_firestore_client, mock_storage_client
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

    # Build mock Storage object
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_blob = mock.create_autospec(storage.Blob)
    predictions = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    mock_blob.download_as_text.return_value = json.dumps(predictions)
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.bucket.return_value = mock_bucket

    # Build mock Firestore document
    mock_document = mock.create_autospec(firestore.DocumentSnapshot)
    mock_document.exists = True
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
    mock_document.to_dict.return_value = metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_document
        )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Chunk "chunk-id" is missing one or more required '
        "fields: row_count, col_count, x_ll_corner, y_ll_corner"
        in str(exc_info.value)
    )


@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions(
    mock_firestore_client, mock_storage_client
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

    # Build mock Storage object
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_blob = mock.create_autospec(storage.Blob)
    predictions = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    mock_blob.download_as_text.return_value = json.dumps(predictions)
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.bucket.return_value = mock_bucket

    # Build mock Firestore document
    mock_document = mock.create_autospec(firestore.DocumentSnapshot)
    mock_document.exists = True
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
    mock_document.to_dict.return_value = metadata
    mock_firestore_client.return_value.collection.return_value.document \
        .return_value.get.return_value = (
            mock_document
        )

    # Build expected output data
    expected_x_coods = np.array([505, 515, 525, 505, 515, 525])
    expected_y_coods = np.array([105, 105, 105, 115, 115, 115])
    expected_gdf_src_crs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(expected_x_coods, expected_y_coods),
        crs="EPSG:32618",
    )
    expected_gdf_global_crs = expected_gdf_src_crs.to_crs("EPSG:4326")
    expected_predictions = [4, 5, 6, 1, 2, 3]
    expected_df = pd.DataFrame(
        {
            "lat": expected_gdf_global_crs.geometry.y,
            "lon": expected_gdf_global_crs.geometry.x,
            "prediction": expected_predictions,
        }
    )

    actual_df = main.export_model_predictions(event)
    assert actual_df.equals(expected_df)
