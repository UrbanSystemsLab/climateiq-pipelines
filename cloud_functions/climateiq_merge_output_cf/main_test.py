from cloudevents import http

import main


def test_merge_scenario_predictions() -> None:
    cloud_event = http.CloudEvent(
        {"type": "google.cloud.storage.object.v1.finalized", "source": "source"}, {}
    )
    assert main.merge_scenario_predictions(cloud_event) is None
