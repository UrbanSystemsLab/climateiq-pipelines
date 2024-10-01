import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from google.cloud import storage

# Main function for the Cloud Function
def aggregate_and_clip_tiff(event, context):
    # Configuration (hardcoded in the function)
    bucket_name = "citycat-layers-dashboard"  # Bucket name for both input and output
    input_prefix = "2m_tiff_output"  # Input prefix where 2m TIFF files are stored
    output_prefix = "10m_clipped_tiff_output"  # Output prefix where aggregated/clipped TIFF files will be stored
    shapefile_path = "/tmp/clipping_boundary.shp"  # Temporary path for shapefile

    # Initialize GCS client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Extract the uploaded TIFF file name from the event
    tiff_file_name = event['name']

    # Only process files from the specified input prefix
    if not tiff_file_name.startswith(input_prefix) or not tiff_file_name.endswith('.tif'):
        print(f"Skipping file: {tiff_file_name}")
        return

    # Download the 2m TIFF file from GCS to a temporary directory
    tiff_file_path = f"/tmp/{os.path.basename(tiff_file_name)}"
    blob = bucket.blob(tiff_file_name)
    blob.download_to_filename(tiff_file_path)

    # Download the shapefile from GCS (assuming it's always required)
    shapefile_prefix = "shapefile/clipping_boundary"
    shapefile_components = ['.shp', '.shx', '.dbf', '.prj']
    for ext in shapefile_components:
        blob_name = f"{shapefile_prefix}{ext}"
        blob = bucket.blob(blob_name)
        blob.download_to_filename(shapefile_path.replace('.shp', ext))

    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Aggregate the 2m TIFF to 10m
    aggregated_tiff_path = tiff_file_path.replace(input_prefix, output_prefix).replace(".tif", "_10m.tif")
    with rasterio.open(tiff_file_path) as src:
        # Calculate the transform and dimensions for the 10m resolution
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
            'compress': 'lzw'
        })

        # Write the aggregated TIFF to temporary storage
        with rasterio.open(aggregated_tiff_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.average)

    print(f"Aggregated TIFF saved to: {aggregated_tiff_path}")

    # Clip the aggregated TIFF using the shapefile boundary
    clipped_tiff_path = aggregated_tiff_path.replace("_10m.tif", "_10m_clipped.tif")
    with rasterio.open(aggregated_tiff_path) as src:
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            'compress': 'lzw'
        })

        # Write the clipped TIFF to temporary storage
        with rasterio.open(clipped_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Clipped TIFF saved to: {clipped_tiff_path}")

    # Upload the aggregated and clipped TIFF back to GCS
    aggregated_blob = bucket.blob(aggregated_tiff_path.replace('/tmp/', f'{output_prefix}/'))
    aggregated_blob.upload_from_filename(aggregated_tiff_path)

    clipped_blob = bucket.blob(clipped_tiff_path.replace('/tmp/', f'{output_prefix}/'))
    clipped_blob.upload_from_filename(clipped_tiff_path)

    print(f"Uploaded aggregated TIFF to {aggregated_blob.name}")
    print(f"Uploaded clipped TIFF to {clipped_blob.name}")

    # Clean up temporary files
    os.remove(tiff_file_path)
    os.remove(aggregated_tiff_path)
    os.remove(clipped_tiff_path)
    for ext in shapefile_components:
        os.remove(shapefile_path.replace('.shp', ext))