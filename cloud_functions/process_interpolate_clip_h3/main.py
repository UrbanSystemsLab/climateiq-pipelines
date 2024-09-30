# Import required libraries
from google.cloud import storage
import gcsfs
import pandas as pd
import h3
import numpy as np
import xarray as xr
import rioxarray as rxr
import os
import tempfile
from shapely.geometry import mapping
import geopandas as gpd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from flask import Request

# Initialize the GCP storage client
client = storage.Client()

# Cloud Function Entry Point
def process_interpolate_clip_h3(request: Request):
    """Cloud function to interpolate rasters, clip using boundary shapefile, and convert to H3 aggregation"""
    # Define GCP bucket names, project ID, paths, and configurations
    source_bucket_name = 'test_dashboard2'
    output_bucket_name = 'test_dashboard2'
    project_id = 'claimateiq-test-2'
    input_file_prefix = '500m_heat_days_'
    output_file_prefix_10m = '10m_disagg_heat_days_'
    path_prefix = 'WRF_ETL/500m_raster_dashboard_layers/'
    output_path_prefix_10m = 'WRF_ETL/10m_raster_dashboard_layers/'
    output_path_prefix_clipped = '10m_clipped'
    output_path_prefix_h3 = 'h3_dashboard'
    shapefile_base_path = 'NYC_boundary/geo_export_e8631fe8-f72a-4bae-ad69-2572187f9018'

    # Step 1: Interpolation from 500m to 10m
    def interp_out_grid(ds, numpoints=10000, method='linear'):
        # Create new coordinates for interpolation
        newx = np.linspace(ds['x'].min().values, ds['x'].max().values, num=numpoints)
        newy = np.linspace(ds['y'].min().values, ds['y'].max().values, num=numpoints)

        # Create template dataset with the new coordinates
        newds = xr.Dataset(coords={'band': np.array([1]),
                                   'x': ('x', newx),
                                   'y': ('y', newy)})

        # Interpolate the data
        ds_interp = ds.interp_like(newds, method=method)
        ds_interp.attrs = ds.attrs
        ds_interp.name = ds.name

        return ds_interp.astype('int32')

    def process_and_interpolate_files():
        bucket = client.bucket(source_bucket_name)
        blobs = bucket.list_blobs(prefix=path_prefix)

        fs = gcsfs.GCSFileSystem(project=project_id)

        for blob in blobs:
            if not blob.name.endswith('.tif') or not blob.name.startswith(input_file_prefix):
                continue

            print(f'Processing {blob.name} for interpolation')
            input_file_path = f'gs://{source_bucket_name}/{blob.name}'
            output_file_name = f'{output_file_prefix_10m}{os.path.basename(blob.name).replace(input_file_prefix, "")}'
            output_file_path = os.path.join(output_path_prefix_10m, output_file_name)

            with fs.open(input_file_path, mode='rb') as f:
                dsin = rxr.open_rasterio(f)
                dsout = interp_out_grid(dsin)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                    temp_filename = temp_file.name
                    dsout.rio.to_raster(temp_filename)

                blob_out = bucket.blob(output_file_path)
                blob_out.upload_from_filename(temp_filename)
                os.remove(temp_filename)
                print(f'Saved interpolated file to {output_file_path}')

    # Step 2: Clipping the rasters using NYC boundary (excluding water)
    def download_shapefile(base_path, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        shapefile_components = ['.shp', '.shx', '.dbf', '.prj']
        for ext in shapefile_components:
            blob_name = f"{base_path}{ext}"
            local_path = os.path.join(local_dir, os.path.basename(blob_name))
            bucket = client.bucket(source_bucket_name)
            blob = bucket.blob(blob_name)
            print(f"Downloading {blob_name} to {local_path}")
            blob.download_to_filename(local_path)
        return os.path.join(local_dir, os.path.basename(base_path) + '.shp')

    def clip_raster_to_boundary(raster, boundary):
        try:
            if raster.rio.crs != boundary.crs:
                raster = raster.rio.reproject(boundary.crs)

            raster.rio.write_nodata(0, inplace=True)
            clipped = raster.rio.clip(boundary.geometry.apply(mapping), boundary.crs, drop=False, invert=False)
            return clipped
        except Exception as e:
            print(f"Error during clipping: {e}")
            return None

    def process_and_clip_files(boundary):
        bucket = client.bucket(output_bucket_name)
        blobs = bucket.list_blobs(prefix=output_path_prefix_10m)

        fs = gcsfs.GCSFileSystem(project=project_id)

        for blob in blobs:
            if not blob.name.endswith('.tif'):
                continue

            print(f'Processing {blob.name} for clipping')
            input_file_path = f'gs://{output_bucket_name}/{blob.name}'
            output_file_name = f'clipped_{os.path.basename(blob.name)}'
            output_file_path = os.path.join(output_path_prefix_clipped, output_file_name)

            with fs.open(input_file_path, mode='rb') as f:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                    temp_filename = temp_file.name
                    temp_file.write(f.read())

                raster = rxr.open_rasterio(temp_filename)
                clipped_raster = clip_raster_to_boundary(raster, boundary)

                if clipped_raster is not None:
                    clipped_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
                    clipped_temp_filename = clipped_temp_file.name
                    clipped_raster.rio.to_raster(clipped_temp_filename, compress='LZW')

                    blob_out = bucket.blob(output_file_path)
                    blob_out.upload_from_filename(clipped_temp_filename)
                    os.remove(temp_filename)
                    os.remove(clipped_temp_filename)
                    print(f'Saved clipped file to {output_file_path}')

    # Step 3: Convert clipped raster to H3 level aggregation
    def raster_to_h3(raster_path, h3_level):
        raster = rxr.open_rasterio(raster_path)
        transform = raster.rio.transform()
        h3_values = defaultdict(list)

        for y in range(raster.rio.height):
            for x in range(raster.rio.width):
                value = raster[0, y, x].item()
                if not np.isnan(value):
                    lon, lat = transform * (x + 0.5, y + 0.5)
                    h3_index = h3.geo_to_h3(lat, lon, h3_level)
                    h3_values[h3_index].append(value)

        h3_data = [[h3_index, np.mean(values)] for h3_index, values in h3_values.items()]
        return pd.DataFrame(h3_data, columns=['h3index', 'value'])

    def process_and_convert_to_h3():
        bucket = client.bucket(output_bucket_name)
        blobs = bucket.list_blobs(prefix=output_path_prefix_clipped)

        fs = gcsfs.GCSFileSystem(project=project_id)

        for blob in blobs:
            if not blob.name.endswith('.tif'):
                continue

            print(f'Processing {blob.name} for H3 conversion')
            input_file_path = f'gs://{output_bucket_name}/{blob.name}'
            output_file_name = f'h3_{os.path.basename(blob.name).replace("clipped_", "")}.csv'
            output_file_path = os.path.join(output_path_prefix_h3, output_file_name)

            with fs.open(input_file_path, mode='rb') as f:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                    temp_filename = temp_file.name
                    temp_file.write(f.read())

                df_h3 = raster_to
