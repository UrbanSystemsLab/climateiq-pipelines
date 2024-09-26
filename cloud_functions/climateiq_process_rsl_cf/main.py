import os
import pandas as pd
from google.cloud import storage
from io import StringIO

def process_rsl_files(request):
    # Get environment variables
    bucket_name = os.getenv('BUCKET_NAME')
    prefix = os.getenv('PREFIX')

    # Initialize the storage client
    client = storage.Client()

    # Get the bucket and list blobs
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    rsl_files = [blob for blob in blobs if blob.name.endswith('.rsl')]

    # Create a dictionary to store max depths
    max_depths = {}

    # Process each .rsl file
    for blob in rsl_files:
        print(f"Processing file: {blob.name}")

        # Read the data into a DataFrame
        data = blob.download_as_string()
        df = pd.read_csv(StringIO(data.decode('utf-8')), delim_whitespace=True, skiprows=1,
                         names=["XCen", "YCen", "Depth", "Vx", "Vy"])

        for _, row in df.iterrows():
            key = (row["XCen"], row["YCen"])
            if key not in max_depths:
                max_depths[key] = row["Depth"]
            else:
                max_depths[key] = max(max_depths[key], row["Depth"])

    # Save results to a CSV file in the same location
    output_file = f"{prefix}25y_6hr_maximum_depth.csv"  # Change the name as needed

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

    max_depth_df = pd.DataFrame(max_depth_data)

    # Convert DataFrame to CSV
    csv_data = max_depth_df.to_csv(index=False)

    # Save the CSV to the bucket
    output_blob = bucket.blob(output_file)
    output_blob.upload_from_string(csv_data, content_type='text/csv')

    return f"Processed {len(rsl_files)} RSL files."