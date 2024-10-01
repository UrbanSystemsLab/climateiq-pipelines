import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np
from google.cloud import storage
from h3 import h3

# Main function for the Cloud Function
def convert_csv_to_geotiff(event, context):
    # Configuration (hardcoded in the function)
    bucket_name = "citycat-layers-dashboard"  # Bucket name for both input and output
    input_prefix = "Max-Depth-csv"  # Input prefix where CSV files are stored
    output_prefix = "2m_tiff_output"  # Output prefix where GeoTIFF files will be stored

    # Initialize GCS client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Extract the uploaded CSV file name from the event
    csv_file_name = event['name']

    # Only process files from the specified input prefix
    if not csv_file_name.startswith(input_prefix) or not csv_file_name.endswith('.csv'):
        print(f"Skipping file: {csv_file_name}")
        return

    # Download the CSV file from GCS to a temporary directory
    csv_file_path = f"/tmp/{os.path.basename(csv_file_name)}"
    blob = bucket.blob(csv_file_name)
    blob.download_to_filename(csv_file_path)

    # Read CSV to DataFrame
    df = pd.read_csv(csv_file_path, on_bad_lines='skip')

    # Convert the CSV data into a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.XCen, df.YCen), crs='EPSG:32618')

    # Define resolution and bounds for the raster
    x_min, x_max = gdf.geometry.x.min(), gdf.geometry.x.max()
    y_min, y_max = gdf.geometry.y.min(), gdf.geometry.y.max()
    pixel_size = 2  # 2m resolution
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    depth_array = np.full((y_res, x_res), np.nan)

    # Fill raster array with depth values from GeoDataFrame
    for i, row in gdf.iterrows():
        x_index = int((row.geometry.x - x_min) / pixel_size)
        y_index = int((y_max - row.geometry.y) / pixel_size)
        if 0 <= x_index < x_res and 0 <= y_index < y_res:
            depth_array[y_index, x_index] = row['Max_Depth']

    # Define output TIFF path
    output_file_name = csv_file_name.replace(input_prefix, output_prefix).replace('.csv', '.tif')
    output_tiff_path = f"/tmp/{os.path.basename(output_file_name)}"

    # Write GeoTIFF to local temporary storage
    with rasterio.open(
        output_tiff_path,
        'w',
        driver='GTiff',
        height=y_res,
        width=x_res,
        count=1,
        dtype=depth_array.dtype,
        crs='EPSG:32618',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(depth_array, 1)

    # Upload the GeoTIFF back to GCS
    output_blob = bucket.blob(output_file_name)
    output_blob.upload_from_filename(output_tiff_path)
    print(f"Uploaded GeoTIFF to {output_blob.name}")

    # Clean up temporary files
    os.remove(csv_file_path)
    os.remove(output_tiff_path)
