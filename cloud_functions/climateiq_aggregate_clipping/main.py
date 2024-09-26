import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from google.cloud import storage

def aggregate_and_clip(request):
    # Get environment variables
    bucket_name = os.getenv('BUCKET_NAME')
    input_prefix = os.getenv('INPUT_PREFIX')
    output_prefix = os.getenv('OUTPUT_PREFIX')
    shapefile_bucket_name = os.getenv('SHAPEFILE_BUCKET_NAME')
    shapefile_path = os.getenv('SHAPEFILE_PATH')

    # Initialize the storage client
    client = storage.Client()

    # Get the bucket and list blobs
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=input_prefix))

    # Download the shapefile
    shapefile_bucket = client.bucket(shapefile_bucket_name)
    shapefile_blob = shapefile_bucket.blob(shapefile_path)
    shapefile_local_path = '/tmp/Manhattan_WGS84.shp'
    shapefile_blob.download_to_filename(shapefile_local_path)

    # Read the shapefile
    shapefile = gpd.read_file(shapefile_local_path)

    for blob in blobs:
        if blob.name.endswith('.tif'):
            input_tiff_path = '/tmp/' + os.path.basename(blob.name)
            blob.download_to_filename(input_tiff_path)

            output_tiff_path = input_tiff_path.replace('.tif', '_10m.tif')
            clipped_tiff_path = input_tiff_path.replace('.tif', '_clipped.tif')

            with rasterio.open(input_tiff_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, 
                    left=src.bounds.left, bottom=src.bounds.bottom, 
                    right=src.bounds.right, top=src.bounds.top, 
                    dst_width=src.width // 5, dst_height=src.height // 5)
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': src.crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw'  # Specify LZW compression
                })

                with rasterio.open(output_tiff_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=src.crs,
                            resampling=Resampling.average)

            with rasterio.open(output_tiff_path) as src:
                out_image, out_transform = mask(src, shapefile.geometry, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform,
                                 'compress': 'lzw'})  # Specify LZW compression

                with rasterio.open(clipped_tiff_path, "w", **out_meta) as dest:
                    dest.write(out_image)

            output_blob = bucket.blob(output_prefix + '/' + os.path.basename(clipped_tiff_path))
            output_blob.upload_from_filename(clipped_tiff_path)
            print(f"Uploaded {clipped_tiff_path} to {output_blob.name}")

            os.remove(input_tiff_path)
            os.remove(output_tiff_path)
            os.remove(clipped_tiff_path)

    return f"Processed {len(blobs)} TIFF files."