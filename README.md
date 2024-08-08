# climateiq-frontend
ClimateIQ Frontend Workstream

Google Fellows: skeshive, sylmak

This repo contains the code for the Apigee Proxy Endpoints and the Export Pipeline ETL.

## Deployment

### apigee

Edit through the [console](
    https://console.cloud.google.com/apigee/proxies?project=climateiq).

### cloud_functions

* **export_to_aws_cf**: [Terraform](
    https://github.com/UrbanSystemsLab/climateiq-terraform)
* **merge_scenario_predictions**:
```
gcloud builds submit --tag us-central1-docker.pkg.dev/climateiq/cloud-run-containers/merge_scenario_predictions:latest
```
* **spatialize_chunk_predictions**: [Terraform](
    https://github.com/UrbanSystemsLab/climateiq-terraform)
* **trigger_export_pipeline**:
```
gcloud builds submit --tag us-central1-docker.pkg.dev/climateiq/cloud-run-containers/trigger-export-pipeline:latest
```

Replace `climateiq` with the appropriate project_id as needed.
