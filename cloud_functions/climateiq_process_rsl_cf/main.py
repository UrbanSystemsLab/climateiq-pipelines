import os
import pandas as pd
from google.cloud import storage
from io import StringIO
import functions_framework

# Trigger function to be called when the function is invoked manually
@functions_framework.http
def process_rsl_files(request):
    """Processes .rsl files in a Google Cloud Storage bucket and saves the max depth to a CSV.
    
    Args:
        request (flask.Request): The request object containing information about the HTTP request.
    
    Returns:
        flask.Response: The response object containing status information.
    """
    # Set your GCP project name, bucket, and prefix for the files
    project_id = "climateiq"
    bucket_name = "climateiq-flood-simulation-output"
    prefix = "Manhattan/config_v1/Rainfall_Data_16.txt/"  # Change folder path in the bucket
    
    # Initialize GCP storage client
    client = storage.Client(project=project_id)

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all .rsl files in the folder
    blobs = list(bucket.list_blobs(prefix=prefix))
    rsl_files = [blob for blob in blobs if blob.name.endswith('.rsl')]

    # Create a dictionary to store max depths
    max_depths = {}

    # Process each .rsl file
    for blob in rsl_files:
        print(f"Processing file: {blob.name}")

        # Download content as string and convert to DataFrame
        data = blob.download_as_string()
        df = pd.read_csv(StringIO(data.decode('utf-8')), delim_whitespace=True, skiprows=1,
                         names=["XCen", "YCen", "Depth", "Vx", "Vy"])

        # Update max depths
        for _, row in df.iterrows():
            key = (row["XCen"], row["YCen"])
            if key not in max_depths:
                max_depths[key] = row["Depth"]
            else:
                max_depths[key] = max(max_depths[key], row["Depth"])

    # Save results to a CSV file in the same location
    output_file = f"{prefix}25y_6hr_maximum_depth.csv"  # Change the output filename if necessary

    print(f"Saving maximum depths to: {output_file}")

    # Prepare data for saving
    max_depth_data = {
        "XCen": [],
        "YCen": [],
        "Max_Depth": []
    }

    for (xcen, ycen), depth in max_depths.items():
        max_depth_data["XCen"].append(xcen)
        max_depth_data["YCen"].append(ycen)
        max_depth_data["Max_Depth"].append(depth)

    # Convert data to DataFrame and then to CSV
    max_depth_df = pd.DataFrame(max_depth_data)
    csv_data = max_depth_df.to_csv(index=False)

    # Save the CSV to the bucket
    output_blob = bucket.blob(output_file)
    output_blob.upload_from_string(csv_data, content_type='text/csv')

    return "Processing completed. CSV saved successfully.", 200
