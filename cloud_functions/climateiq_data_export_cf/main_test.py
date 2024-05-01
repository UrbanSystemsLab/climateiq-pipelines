import main
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd

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
def test_export_model_predictions_missing_study_area(
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
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.return_value = (
        mock_study_area
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Study area "study-area-name" does not exist' in str(exc_info.value)


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
    mock_study_area.exists = True
    expected_metadata = {
        "name": "study_area_name",
        "crs": "EPSG:32618",
        "chunks": {},  # Missing "cell_size" required field
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.return_value = (
        mock_study_area
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Study area "study-area-name" is missing one or more required field(s): cell_size, crs, chunks'
        in str(exc_info.value)
    )


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_missing_chunk(
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
        "cell_size": 10,
        "chunks": {"missing-chunk-id": {"col_count": 10, "row_count": 5}},
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.return_value = (
        mock_study_area
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert 'Chunk "chunk-id" does not exist' in str(exc_info.value)


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_export_model_predictions_invalid_chunk(
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
        "cell_size": 10,
        "crs": "EPSG:32618",
        "chunks": {
            "chunk-id": {
                "col_count": 3,
                "x_ll_corner": 500,
                "y_ll_corner": 100,
            }
        },  # Missing "row_count" required field
    }
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.return_value = (
        mock_study_area
    )

    with pytest.raises(ValueError) as exc_info:
        main.export_model_predictions(event)

    assert (
        'Chunk "chunk-id" is missing one or more required fields: row_count, col_count, x_ll_corner, y_ll_corner'
        in str(exc_info.value)
    )


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
    mock_study_area.to_dict.return_value = expected_metadata
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.return_value = (
        mock_study_area
    )

    # Build expected output data
    expected_x_centers = np.array([505, 515, 525, 505, 515, 525])
    expected_y_centers = np.array([105, 105, 105, 115, 115, 115])
    expected_gdf_src_crs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(expected_x_centers, expected_y_centers),
        crs="EPSG:32618",
    )
    expected_gdf_global_crs = expected_gdf_src_crs.to_crs("EPSG:4326")
    expected_df = pd.DataFrame(
        {
            "lat": expected_gdf_global_crs.geometry.y,
            "lon": expected_gdf_global_crs.geometry.x,
        }
    )

    actual_df = main.export_model_predictions(event)
    assert actual_df.equals(expected_df)
