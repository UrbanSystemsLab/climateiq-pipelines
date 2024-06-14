from cloudevents import http
import functions_framework


@functions_framework.cloud_event
def merge_scenario_predictions(cloud_event: http.CloudEvent) -> None:
    return
