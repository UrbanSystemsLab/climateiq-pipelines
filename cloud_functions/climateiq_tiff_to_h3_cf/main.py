import os
import pandas as pd
import rasterio
import numpy as np
from h3 import h3
from google.cloud import storage
from shapely.geometry import Polygon

# Main function for the Cloud Function
def convert_tiff_to_h3_csv(event, context):
    # Configuration (hardcoded in the function)
    bucket_name = "citycat-layers-dashboard"  # Bucket name for both input and output
    input_prefix = "10m_clipped_tiff_output"  # Input prefix where 10m clipped TIFF files are stored
    output_prefix = "h3_csv_output"  # Output prefix where H3 index CSV files will be stored
    h3_resolution = 13  # H3 resolution level

    # Initialize GCS client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Extract the uploaded TIFF file name from the event
    tiff_file_name = event['name']

    # Only process files from the specified input prefix
    if not tiff_file_name.startswith(input_prefix) or not tiff_file_name.endswith('.tif'):
        print(f"Skipping file: {tiff_file_name}")
        return

    # Download the 10m clipped TIFF file from GCS to a temporary directory
    tiff_file_path = f"/tmp/{os.path.basename(tiff_file_name)}"
    blob = bucket.blob(tiff_file_name)
    blob.download_to_filename(tiff_file_path)

    # Function to read raster using rasterio
    def read_raster(file_path):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read the first band
                profile = src.profile  # Get metadata about the raster file
                nodata = src.nodata  # Get the nodata value
            return data, profile, nodata
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading file {file_path}: {e}")
            return None, None, None

    # Read the GeoTIFF file
    data, profile, nodata = read_raster(tiff_file_path)

    if data is not None and profile is not None:
        # Define the pixel size (in degrees)
        pixel_size_x = profile['transform'][0]
        pixel_size_y = -profile['transform'][4]

        # Create a DataFrame to store pixel coordinates and values
        data_list = []
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if value != nodata and not np.isnan(value):  # Exclude nodata and NaN values
                    x, y = profile['transform'] * (col + 0.5, row + 0.5)  # Get the center coordinates of the pixel
                    data_list.append([y, x, value])

        df = pd.DataFrame(data_list, columns=['latitude', 'longitude', 'value'])

        # Function to sample points within a pixel
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

        # Function to find overlapping H3 cells for sampled points
        def assign_pixels_to_h3(df, res, pixel_size_x, pixel_size_y, num_samples=5):
            h3_data = []
            for _, row in df.iterrows():
                lat, lon, value = row['latitude'], row['longitude'], row['value']
                sampled_points = sample_points_within_pixel(lat, lon, pixel_size_x, pixel_size_y, num_samples)
                for point in sampled_points:
                    h3_cell = h3.geo_to_h3(point[0], point[1], res)
                    h3_data.append([h3_cell, value])
            return pd.DataFrame(h3_data, columns=['cell_code', 'value'])

        # Convert DataFrame and assign pixel values to H3 cells
        h3_df = assign_pixels_to_h3(df, h3_resolution, pixel_size_x, pixel_size_y, num_samples=5)

        # Convert 'value' column from feet to meters
        h3_df['value'] = h3_df['value'] * 0.3048

        # Aggregate data into H3 using the mean
        h3_aggregated = h3_df.groupby('cell_code')['value'].mean().reset_index()

        # Save the aggregated H3 data to CSV in temporary storage
        output_csv_path = tiff_file_path.replace('.tif', '.csv')
        h3_aggregated.to_csv(output_csv_path, index=False)
        print(f"Saved H3 data to {output_csv_path}")

        # Upload the CSV to GCS
        output_blob = bucket.blob(output_csv_path.replace('/tmp/', f'{output_prefix}/'))
        output_blob.upload_from_filename(output_csv_path)
        print(f"Uploaded CSV to {output_blob.name}")

        # Clean up temporary files
        os.remove(tiff_file_path)
        os.remove(output_csv_path)
