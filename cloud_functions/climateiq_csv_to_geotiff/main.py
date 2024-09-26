import os
import pandas as pd
from google.cloud import storage
from io import StringIO
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np

def convert_csv_to_geotiff(request):
    # Get environment variables
    bucket_name = os.getenv('BUCKET_NAME')
    prefix = os.getenv('PREFIX')

    # Initialize the storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # WKT for EPSG:32618 (NYC flood data is provided in UTM zone 18 CRS)
    wkt_epsg_32618 = (
        'PROJCS["WGS 84 / UTM zone 18N",'
        'GEOGCS["WGS 84",'
        'DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],'
        'PARAMETER["central_meridian",-75],'
        'PARAMETER["scale_factor",0.9996],'
        'PARAMETER["false_easting",500000],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1,'
        'AUTHORITY["EPSG","9001"]]]'
    )

    # List all files in the bucket with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Process each CSV file in the bucket
    for blob in blobs:
        if blob.name.endswith('.csv'):
            # Download the CSV file from GCS to a temp folder
            csv_file_path = '/tmp/' + os.path.basename(blob.name)
            blob.download_to_filename(csv_file_path)

            # Read the CSV file
            df = pd.read_csv(csv_file_path, on_bad_lines='skip')

            # Convert the CSV data into a GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.XCen, df.YCen), crs='EPSG:32618'
            )

            # Reproject to EPSG:32618 (WGS84)
            gdf = gdf.to_crs(epsg=32618)

            # Check if the CRS is set correctly
            print(f"Processing {blob.name}, CRS: {gdf.crs}")

            # Generate a GeoTIFF from the GeoDataFrame using rasterio
            output_tiff_path = '/tmp/' + os.path.basename(blob.name).replace('.csv', '.tif')
            x_min, x_max = gdf.geometry.x.min(), gdf.geometry.x.max()
            y_min, y_max = gdf.geometry.y.min(), gdf.geometry.y.max()

            # Set resolution of raster
            pixel_size = 2  # 2 meter resolution
            x_res = int((x_max - x_min) / pixel_size)
            y_res = int((y_max - y_min) / pixel_size)

            transform = from_origin(x_min, y_max, pixel_size, pixel_size)
            depth_array = np.full((y_res, x_res), np.nan)

            for i, row in gdf.iterrows():
                x_index = int((row.geometry.x - x_min) / pixel_size)
                y_index = int((y_max - row.geometry.y) / pixel_size)

                # Check if indices are within bounds
                if 0 <= x_index < x_res and 0 <= y_index < y_res:
                    depth_array[y_index, x_index] = row['Max_Depth']

            with rasterio.open(
                output_tiff_path,
                'w',
                driver='GTiff',
                height=y_res,
                width=x_res,
                count=1,
                dtype=depth_array.dtype,
                crs=wkt_epsg_32618,
                transform=transform,
            ) as dst:
                dst.write(depth_array, 1)

            print(f"Finished processing {blob.name}. Output saved to {output_tiff_path}")

            # Upload the GeoTIFF to the GCS bucket
            output_blob = bucket.blob(prefix + '/' + os.path.basename(output_tiff_path))
            output_blob.upload_from_filename(output_tiff_path)
            print(f"Uploaded {output_tiff_path} to {output_blob.name}")

            # Remove temporary files
            os.remove(csv_file_path)
            os.remove(output_tiff_path)

    return f"Processed {len(blobs)} CSV files."