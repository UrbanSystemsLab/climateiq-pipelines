from google.cloud import storage
import gcsfs
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from h3 import h3
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import tempfile
import os
from flask import Request

# Initialize the GCP storage client
client = storage.Client()

# Cloud Function Entry Point
def process_raster_to_h3_and_merge_csv(request: Request):
    """Cloud function to process rasters into H3 cells and merge CSV files for heat data"""
    # Define GCP project, bucket, and paths
    project_id = 'claimateiq-test-2'
    bucket_name = 'wrf_dashboard_layers'
    path_prefix = 'WRF_ETL/clipped_10m_raster_dashboard_layers/'
    output_csv_path = 'gs://wrf_dashboard_layers/WRF_ETL/clipped_10m_h3_csv_dashboard_layers/merged_NYC_heat_int32.csv'

    # Create the GCS file system object
    fs = gcsfs.GCSFileSystem(project=project_id)

    # Step 1: Read the list of all .tif files
    def list_tif_files(bucket_name, path_prefix):
        bucket = client.bucket(bucket_name)
        tif_files = [blob.name for blob in bucket.list_blobs(prefix=path_prefix) if blob.name.endswith('.tif')]
        return tif_files

    # Step 2: Read a raster from GCS and convert it to H3
    def raster_to_h3(bucket_name, file_path, output_csv_path, resolution=13):
        geotiff_path = f'gs://{bucket_name}/{file_path}'
        data, profile, nodata = read_raster(geotiff_path)

        if data is not None and profile is not None:
            # Define the pixel size (in degrees)
            pixel_size_x = profile['transform'][0]
            pixel_size_y = -profile['transform'][4]

            # Create a DataFrame to store pixel coordinates and values
            data_list = []
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    value = data[row, col]
                    if value != nodata and not np.isnan(value):
                        x, y = profile['transform'] * (col + 0.5, row + 0.5)
                        data_list.append([y, x, value])

            df = pd.DataFrame(data_list, columns=['latitude', 'longitude', 'value'])

            # Convert to H3
            h3_df = assign_pixels_to_h3(df, resolution, pixel_size_x, pixel_size_y, num_samples=5)

            # Aggregate and save
            h3_aggregated = h3_df.groupby('cell_code')['value'].mean().reset_index()
            h3_aggregated['value'] = h3_aggregated['value'].astype(np.int32)

            # Save to CSV
            with fs.open(output_csv_path, 'w') as f:
                h3_aggregated.to_csv(f, index=False)
            print(f"Saved H3 data to {output_csv_path}")

    def read_raster(file_path):
        try:
            with fs.open(file_path, 'rb') as f:
                with rasterio.open(f) as src:
                    data = src.read(1)  # Read the first band
                    profile = src.profile  # Get metadata about the raster file
                    nodata = src.nodata  # Get the nodata value
            return data, profile, nodata
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading file {file_path}: {e}")
            return None, None, None

    def assign_pixels_to_h3(df, res, pixel_size_x, pixel_size_y, num_samples=5):
        h3_data = []
        for _, row in df.iterrows():
            lat, lon, value = row['latitude'], row['longitude'], row['value']
            sampled_points = sample_points_within_pixel(lat, lon, pixel_size_x, pixel_size_y, num_samples)
            for point in sampled_points:
                h3_cell = h3.geo_to_h3(point[0], point[1], res)
                h3_data.append([h3_cell, value])
        return pd.DataFrame(h3_data, columns=['cell_code', 'value'])

    def sample_points_within_pixel(lat, lon, pixel_size_x, pixel_size_y, num_samples=5):
        points = []
        lat_step = pixel_size_y / num_samples
        lon_step = pixel_size_x / num_samples
        for i in range(num_samples):
            for j in range(num_samples):
                sample_lat = lat - i * lat_step
                sample_lon = lon + j * lon_step
                points.append((sample_lat, sample_lon))
        return points

    # Step 3: Merge all CSV files into one CSV file
    def merge_csv_files(csv_folder_path, output_csv_path):
        # Initialize an empty DataFrame for the merged result
        merged_df = pd.DataFrame()

        # List all CSV files in the folder
        blobs = fs.ls(f'gs://{bucket_name}/{csv_folder_path}')
        csv_files = [blob for blob in blobs if blob.endswith('.csv')]

        print("Found the following CSV files:")
        for csv_file in csv_files:
            print(csv_file)

        # Loop through all CSV files in the folder
        for csv_file in csv_files:
            csv_path = f'gs://{csv_file}'

            try:
                # Read each CSV file
                with fs.open(csv_path, 'r') as f:
                    df = pd.read_csv(f)

                # Convert all int64 columns to int32
                for col in df.columns:
                    if pd.api.types.is_integer_dtype(df[col]) and df[col].dtype == 'int64':
                        df[col] = df[col].astype('int32')

                # Merge with the main DataFrame on 'cell_code'
                if merged_df.empty:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on='cell_code', how='outer')

            except FileNotFoundError as e:
                print(f"File not found: {csv_file}")
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")

        # Save the merged DataFrame back to GCS
        if not merged_df.empty:
            with fs.open(output_csv_path, 'w') as f:
                merged_df.to_csv(f, index=False)
            print(f"Merged CSV saved to {output_csv_path}")
        else:
            print("No consistent CSV files found for merging.")

    # Execute all steps
    try:
        # List all tif files
        tif_files = list_tif_files(bucket_name, path_prefix)
        
        # Process each tif file to H3 and save CSV
        for tif_file in tif_files:
            output_csv_path_individual = f'gs://wrf_dashboard_layers/WRF_ETL/clipped_10m_h3_csv_dashboard_layers/{os.path.basename(tif_file)}.csv'
            raster_to_h3(bucket_name, tif_file, output_csv_path_individual)

        # Merge all the individual CSV files into one
        merge_csv_files('WRF_ETL/clipped_10m_h3_csv_dashboard_layers/', output_csv_path)

    except Exception as e:
        print(f"Error during processing: {e}")

    return "Processing Completed"
