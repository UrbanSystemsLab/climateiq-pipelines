from google.cloud import storage
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.transform import from_origin
from affine import Affine
import gcsfs
import os
import tempfile

# Initialize the GCP storage client
client = storage.Client()

# Function to count the number of days with a variable (T2, HI, RH2) above a threshold
def count_days_above_threshold(bucket_name, path_prefix, thresholds, variable, output_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path_prefix)

    fs = gcsfs.GCSFileSystem(project='claimateiq-test-2')

    count_above_thresholds = {threshold: None for threshold in thresholds}
    lons, lats = None, None

    for blob in blobs:
        if not blob.name.endswith('.nc'):
            continue

        with fs.open(f'gs://{bucket_name}/{blob.name}', mode='rb') as f:
            nc = Dataset('in_memory', memory=f.read())

            data = nc.variables[variable][:]
            if variable != 'RH2_max':
                data_celsius = data - 273.15  # Convert from Kelvin to Celsius
            else:
                data_celsius = data

            for threshold in thresholds:
                if count_above_thresholds[threshold] is None:
                    count_above_thresholds[threshold] = np.zeros_like(data_celsius, dtype=int)
                    lons = nc.variables['XLONG'][0, :, :]
                    lats = nc.variables['XLAT'][0, :, :]

                count_above_thresholds[threshold] += (data_celsius > threshold).astype(int)

            nc.close()

    for threshold, count_above_threshold in count_above_thresholds.items():
        output_file_name = f'{output_prefix}_{threshold}c.tif'
        save_days_above_threshold_to_geotiff(count_above_threshold, lons, lats, output_file_name, bucket_name)

# Function to save the days above a threshold to a GeoTIFF file
def save_days_above_threshold_to_geotiff(count_above_threshold, lons, lats, output_file_path, output_bucket_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
        temp_filename = temp_file.name

        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        pixel_width = (lon_max - lon_min) / lons.shape[1]
        pixel_height = (lat_max - lat_min) / lats.shape[0]
        transform = Affine.translation(lon_min, lat_max) * Affine.scale(pixel_width, -pixel_height)

        flipped_data = np.flipud(count_above_threshold[0, :, :])

        with rasterio.open(
            temp_filename, 'w', driver='GTiff',
            height=flipped_data.shape[0], width=flipped_data.shape[1],
            count=1, dtype='int32', crs='+proj=latlong',
            transform=transform) as dst:
            dst.write(flipped_data, 1)

    # Upload the temporary file to GCS
    client = storage.Client()
    bucket = client.bucket(output_bucket_name)
    blob = bucket.blob(output_file_path)
    blob.upload_from_filename(temp_filename)
    os.remove(temp_filename)
    print(f'Saved GeoTIFF file to {output_file_path}')

# Main Cloud Function
def process_wrf_thresholds(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    # Parameters from HTTP request
    output_bucket_name = request_json.get('output_bucket_name') if request_json else request_args.get('output_bucket_name')
    path_prefix = request_json.get('path_prefix') if request_json else request_args.get('path_prefix')
    variable = request_json.get('variable') if request_json else request_args.get('variable')
    thresholds = request_json.get('thresholds') if request_json else request_args.get('thresholds')

    # Set default thresholds if not provided
    if not thresholds:
        thresholds = [25, 30, 35, 40] if variable == 'T2_max' else [30, 35, 40, 45]

    if not output_bucket_name or not path_prefix or not variable:
        return "Missing required parameters: output_bucket_name, path_prefix, variable", 400

    # Run threshold calculation
    count_days_above_threshold(output_bucket_name, path_prefix, thresholds, variable, path_prefix)

    return f"Processed variable {variable} with thresholds {thresholds}", 200
