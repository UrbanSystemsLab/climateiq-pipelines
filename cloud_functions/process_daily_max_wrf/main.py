from google.cloud import storage
import gcsfs
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import wrf
import tempfile
import os
import math
from flask import jsonify

# Initialize the GCP storage client
client = storage.Client()

def wrf_etl(request):
    """HTTP Cloud Function that processes WRF output files for daily max T2, RH2, and Heat Index (HI).

    Args:
        request (flask.Request): The request object. Expected to contain JSON payload with necessary parameters.

    Returns:
        JSON response.
    """
    request_json = request.get_json(silent=True)

    # Extract parameters from the request
    if not request_json:
        return jsonify({"error": "Invalid request. No input parameters provided."}), 400

    source_bucket_name = request_json.get('source_bucket')
    output_bucket_name = request_json.get('output_bucket')
    path_prefix = request_json.get('path_prefix')
    prefix_date_ranges = request_json.get('prefix_date_ranges')

    if not (source_bucket_name and output_bucket_name and path_prefix and prefix_date_ranges):
        return jsonify({"error": "Missing required parameters."}), 400

    # Get the filtered list of files
    files_in_range = list_files_within_date_ranges(source_bucket_name, prefix_date_ranges)

    # Organize files by day
    files_by_day = organize_files_by_day(files_in_range)

    # Process each day's files
    for date_str, day_files in files_by_day.items():
        print(f"Processing files for date: {date_str}")
        process_day_files(day_files, path_prefix, output_bucket_name, date_str)
        print(f"Completed processing for date: {date_str}")

    return jsonify({"status": "Processing completed successfully."})

def list_files_within_date_ranges(bucket_name, prefix_date_ranges):
    """Lists files within given date ranges."""
    bucket = client.bucket(bucket_name)
    filtered_files = []

    for prefix, start_date, end_date in prefix_date_ranges:
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d_%H:%M:%S")
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            try:
                parts = blob.name.split('/')[-1].split('_')
                date_str = parts[2] + '_' + parts[3]
                file_datetime = datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")
                if start_datetime <= file_datetime <= end_datetime:
                    filtered_files.append(blob.name)
            except (IndexError, ValueError) as e:
                print(f"Skipping file {blob.name} due to error: {e}")

    return filtered_files

def organize_files_by_day(files_in_range):
    """Organizes files by day."""
    files_by_day = {}
    for file in files_in_range:
        parts = file.split('/')[-1].split('_')
        date_str = parts[2]
        if date_str not in files_by_day:
            files_by_day[date_str] = []
        files_by_day[date_str].append(file)
    return files_by_day

def process_day_files(day_files, path_prefix, output_bucket_name, date_str):
    """Processes daily files and saves maximum T2, RH2, and Heat Index (HI)."""
    t2_data = []
    rh2_data = []
    fs = gcsfs.GCSFileSystem(project='YOUR_PROJECT_ID')

    for file in day_files:
        with fs.open(f'gs://{source_bucket_name}/{file}', mode='rb') as f:
            nc = Dataset('in_memory', memory=f.read())
            t2 = wrf.getvar(nc, 'T2')
            rh2 = wrf.getvar(nc, 'rh2')
            t2_data.append(t2)
            rh2_data.append(rh2)
            nc.close()

    if t2_data and rh2_data:
        t2_max, rh2_max, hi_max = get_max_values_and_hi(t2_data, rh2_data)
        save_to_netcdf(t2_max, rh2_max, hi_max, day_files[0], output_bucket_name, path_prefix, date_str)

def get_max_values_and_hi(t2_data, rh2_data):
    """Gets max values for T2, RH2, and computes Heat Index (HI) in Kelvin."""
    t2_array = np.array(t2_data)
    rh2_array = np.array(rh2_data)

    # Convert T2 from Kelvin to Fahrenheit for HI computation
    t2_fahrenheit = kelvin_to_fahrenheit(t2_array)

    # Compute maximum temperature and corresponding RH
    max_temp_fahrenheit = np.max(t2_fahrenheit, axis=0)
    indices_of_max_temp = np.argmax(t2_fahrenheit, axis=0)
    corresponding_rh = np.take_along_axis(rh2_array, np.expand_dims(indices_of_max_temp, axis=0), axis=0).squeeze(axis=0)

    # Compute heat index in Fahrenheit
    hi_fahrenheit = np.zeros_like(max_temp_fahrenheit)
    for i in range(max_temp_fahrenheit.shape[0]):
        for j in range(max_temp_fahrenheit.shape[1]):
            hi_fahrenheit[i, j] = compute_heat_index(max_temp_fahrenheit[i, j], corresponding_rh[i, j])

    # Convert heat index to Kelvin
    hi_kelvin = fahrenheit_to_kelvin(hi_fahrenheit)

    # Get max T2 and RH2
    t2_max = np.max(t2_array, axis=0)
    rh2_max = np.max(rh2_array, axis=0)

    return t2_max, rh2_max, hi_kelvin

def compute_heat_index(T, RH):
    """Compute the heat index in Fahrenheit."""
    simple_HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    if simple_HI < 80:
        return simple_HI

    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * RH
        - 0.22475541 * T * RH
        - 0.00683783 * T * T
        - 0.05481717 * RH * RH
        + 0.00122874 * T * T * RH
        + 0.00085282 * T * RH * RH
        - 0.00000199 * T * T * RH * RH
    )

    # Adjustments for low and high humidity
    if RH < 13 and 80 <= T <= 112:
        adjustment = ((13 - RH) / 4) * np.sqrt((17 - abs(T - 95)) / 17)
        HI -= adjustment

    if RH > 85 and 80 <= T <= 87:
        adjustment = ((RH - 85) / 10) * ((87 - T) / 5)
        HI += adjustment

    return HI

def kelvin_to_fahrenheit(temp_k):
    """Convert temperature from Kelvin to Fahrenheit."""
    return (temp_k - 273.15) * 9/5 + 32

def fahrenheit_to_kelvin(temp_f):
    """Convert temperature from Fahrenheit to Kelvin."""
    return (temp_f - 32) * 5/9 + 273.15

def save_to_netcdf(t2_max, rh2_max, hi_max, template_file, output_bucket_name, path_prefix, date_str):
    """Saves the max T2, RH2, and HI values to a NetCDF file."""
    fs = gcsfs.GCSFileSystem(project='YOUR_PROJECT_ID')
    with fs.open(f'gs://{source_bucket_name}/{template_file}', mode='rb') as f:
        template_nc = Dataset('in_memory', memory=f.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as temp_file:
        temp_filename = temp_file.name
        with Dataset(temp_filename, 'w', format='NETCDF4') as nc:
            for name, dimension in template_nc.dimensions.items():
                nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            for name, variable in template_nc.variables.items():
                if name not in ['T2', 'RH2']:
                    out_var = nc.createVariable(name, variable.datatype, variable.dimensions)
                    out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                    out_var[:] = variable[:]
            t2_max_var = nc.createVariable('T2_max', 'f4', ('Time', 'south_north', 'west_east'))
            t2_max_var[:] = t2_max
            rh2_max_var = nc.createVariable('RH2_max', 'f4', ('Time', 'south_north', 'west_east'))
            rh2_max_var[:] = rh2_max
            hi_max_var = nc.createVariable('HI_max', 'f4', ('Time', 'south_north', 'west_east'))
            hi_max_var[:] = hi_max

    # Upload the NetCDF file to GCS
    client = storage.Client()
    bucket = client.bucket(output_bucket_name)
    blob = bucket.blob(f'{path_prefix}{date_str}.nc')
    blob.upload_from_filename(temp_filename)
    os.remove(temp_filename)
    print(f'Saved NetCDF file to {path_prefix}{date_str}.nc')
