import os
import pandas as pd
from google.cloud import storage
import rasterio
import numpy as np
from h3 import h3
from shapely.geometry import Polygon

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

# Function to convert H3 index to a polygon
def h3_to_polygon(h):
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon(boundary)

def convert_tiff_to_h3(request):
    # Get environment variables
    bucket_name = os.getenv('BUCKET_NAME')
    input_prefix = os.getenv('INPUT_PREFIX')
    output_prefix = os.getenv('OUTPUT_PREFIX')

    # Initialize the storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all .tif files in the bucket with the specified prefix
    blobs = list(bucket.list_blobs(prefix=input_prefix))

    for blob in blobs:
        if blob.name.endswith('.tif'):
            # Download the TIFF file from GCS to a temp folder
            tiff_file_path = '/tmp/' + os.path.basename(blob.name)
            blob.download_to_filename(tiff_file_path)

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

                # Convert DataFrame and assign pixel values to H3 cells
                resolution = 13  # H3 level 13
                h3_df = assign_pixels_to_h3(df, resolution, pixel_size_x, pixel_size_y, num_samples=5)

                # Convert 'value' column from feet to meters
                h3_df['value'] = h3_df['value'] * 0.3048

                # Aggregate data into H3 using the mean
                h3_aggregated = h3_df.groupby('cell_code')['value'].mean().reset_index()

                # Save the aggregated H3 data to CSV
                output_csv_path = tiff_file_path.replace('.tif', '.csv')
                h3_aggregated.to_csv(output_csv_path, index=False)
                print(f"Saved H3 data to {output_csv_path}")

                # Upload the CSV to the GCS bucket
                output_blob = bucket.blob(output_prefix + '/' + os.path.basename(output_csv_path))
                output_blob.upload_from_filename(output_csv_path)
                print(f"Uploaded {output_csv_path} to {output_blob.name}")

                # Remove temporary files
                os.remove(tiff_file_path)
                os.remove(output_csv_path)

    return f"Processed {len(blobs)} TIFF files."