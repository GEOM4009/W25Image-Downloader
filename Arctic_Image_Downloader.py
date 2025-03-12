import datetime
from datetime import UTC
import geopandas as gpd
import requests
import earthaccess
import sys
from osgeo import gdal, osr, gdalconst
import os
import numpy as np
import glob

def verify_credentials():
    """
    Verify NASA Earthdata credentials and ensure successful login
    Returns authenticated earthaccess object or exits if authentication fails
    """
    try:
        print("Attempting to authenticate with NASA Earthdata...")
        auth = earthaccess.login()
        
        # Test authentication by trying to search (this will fail if auth is invalid)
        test_search = earthaccess.search_data(
            short_name="MOD02QKM",
            temporal=(
                datetime.datetime.now(UTC) - datetime.timedelta(hours=1),
                datetime.datetime.now(UTC)
            )
        )
        print("✓ Successfully authenticated with NASA Earthdata!")
        return auth
    except Exception as e:
        print("✗ Authentication failed!")
        print("Error:", str(e))
        print("\nPlease check your credentials:")
        print("1. Ensure you have a NASA Earthdata account")
        print("2. Your credentials should be stored in ~/.netrc or _netrc file")
        print("3. You can reset the process by deleting the .netrc file and running again")
        sys.exit(1)

def get_modis_imagery(min_lon, min_lat, max_lon, max_lat, hours_ago=5, resolution="both"):
    """
    Get MODIS imagery for a given bounding box and time range.
    
    Parameters:
    - min_lon, min_lat, max_lon, max_lat: bounding box coordinates
    - hours_ago: how far back to search for imagery
    - resolution: "QKM" (250m), "HKM" (500m), or "both" to download both resolutions
    """
    # Calculate time range using timezone-aware datetime
    end_time = datetime.datetime.now(UTC)
    start_time = end_time - datetime.timedelta(hours=hours_ago)
    
    print(f"\nSearch Parameters:")
    print(f"Time Range: From {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"            To   {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Area of interest: {abs(min_lon):.2f}°W, {min_lat:.2f}°N to {abs(max_lon):.2f}°W, {max_lat:.2f}°N")
    print(f"Resolution: {'Both 250m and 500m' if resolution == 'both' else ('250m' if resolution == 'QKM' else '500m')}")
    
    all_results = []
    result_types = []  # Store the resolution type for each result
    
    # List of products to search based on resolution parameter
    products_to_search = []
    if resolution == "QKM" or resolution == "both":
        products_to_search.append(("MOD02QKM", "MYD02QKM", "250m"))
    if resolution == "HKM" or resolution == "both":
        products_to_search.append(("MOD02HKM", "MYD02HKM", "500m"))
    
    # Search for each product type
    for terra_product, aqua_product, res_desc in products_to_search:
        # Search Terra MODIS
        print(f"\nSearching Terra MODIS ({terra_product} - {res_desc})...")
        terra_results = earthaccess.search_data(
            short_name=terra_product,
            temporal=(start_time, end_time),
            bounding_box=(min_lon, min_lat, max_lon, max_lat)
        )
        if terra_results:
            all_results.extend(terra_results)
            # Add corresponding resolution info for each result
            result_types.extend([terra_product] * len(terra_results))
            print(f"Found {len(terra_results)} Terra {res_desc} images")
        
        # Search Aqua MODIS
        print(f"Searching Aqua MODIS ({aqua_product} - {res_desc})...")
        aqua_results = earthaccess.search_data(
            short_name=aqua_product,
            temporal=(start_time, end_time),
            bounding_box=(min_lon, min_lat, max_lon, max_lat)
        )
        if aqua_results:
            all_results.extend(aqua_results)
            # Add corresponding resolution info for each result
            result_types.extend([aqua_product] * len(aqua_results))
            print(f"Found {len(aqua_results)} Aqua {res_desc} images")
    
    return all_results, result_types

def download_and_process_image(granule, output_filename, product_type):
    """
    Download and process a MODIS image with proper projection for Arctic regions.
    
    Parameters:
    - granule: The DataGranule object to download
    - output_filename: The base filename for the output file
    - product_type: The product type (e.g., "MOD02QKM", "MYD02HKM")
    """
    os.makedirs("./downloads", exist_ok=True)
    
    # Extract resolution from product_type
    resolution = "unknown"
    if "QKM" in product_type:
        resolution = "250m"
    elif "HKM" in product_type:
        resolution = "500m"
    
    # Add resolution to filename
    base_name, ext = os.path.splitext(output_filename)
    resolution_output_filename = f"{base_name}_{resolution}{ext}"
    
    print(f"Downloading {product_type} granule ({resolution})...")
    downloaded_files = earthaccess.download(granule, local_path="./downloads")
    hdf_file = None
    
    if downloaded_files:
        print(f"Processing image...")
        hdf_file = downloaded_files[0]
        
        # Register all GDAL drivers
        gdal.AllRegister()
        
        # Open the HDF file
        hdf_dataset = gdal.Open(hdf_file)
        
        if hdf_dataset is None:
            print(f"Failed to open HDF file: {hdf_file}")
            return
        
        # List all subdatasets
        subdatasets = hdf_dataset.GetSubDatasets()
        
        print("Available subdatasets:")
        for i, subdataset in enumerate(subdatasets):
            print(f"{i + 1}: {subdataset[0]} - {subdataset[1]}")
        
        # Get the first subdataset (EV_250_RefSB for QKM or EV_500_RefSB for HKM)
        subdataset_path = subdatasets[0][0]
        
        # Create output file paths
        temp_output = os.path.join("./downloads", "temp_" + resolution_output_filename)
        final_output = os.path.join("./downloads", resolution_output_filename)
        
        try:
            # Open the subdataset
            subdataset = gdal.Open(subdataset_path)
            if subdataset is None:
                print("Failed to open subdataset")
                return
            
            # Get geolocation datasets (latitude and longitude)
            geoloc_dataset = gdal.Open(hdf_file)
            latitude = geoloc_dataset.GetSubDatasets()[0][0]  # Assuming lat is first
            longitude = geoloc_dataset.GetSubDatasets()[1][0]  # Assuming lon is second
            
            # Create the output dataset with Arctic Polar Stereographic projection
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3995)  # EPSG:3995 - Arctic Polar Stereographic
            
            # Use GDAL Warp
            warp_options = gdal.WarpOptions(
                format='GTiff',
                dstSRS=srs.ExportToWkt(),
                outputType=gdal.GDT_Float32,
                resampleAlg=gdal.GRA_Bilinear,
                geoloc=True,  # Use geolocation arrays
                multithread=True,
                warpOptions=['NUM_THREADS=ALL_CPUS']
            )
            
            # Perform the warping operation
            gdal.Warp(final_output, subdataset, options=warp_options)
            
            print(f"Image processed and saved to {final_output}")
            
            # Verify the output
            verify_projection(final_output)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
        
        finally:
            # Clean up
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            # Close datasets
            hdf_dataset = None
            if 'subdataset' in locals() and subdataset:
                subdataset = None
            if 'geoloc_dataset' in locals() and geoloc_dataset:
                geoloc_dataset = None
            
    else:
        print("Failed to download image.")
    
    # Return the HDF file path for later cleanup
    return hdf_file

def verify_projection(filename):
    """
    Verify the projection of a processed image.
    """
    try:
        dataset = gdal.Open(filename)
        if dataset is None:
            print(f"Could not open {filename}")
            return
            
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        
        print(f"\nProjection Information for {filename}:")
        print(f"Projection: {projection}")
        print(f"Geotransform: {geotransform}")
        
        # Get raster dimensions
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        print(f"Dimensions: {width}x{height}")
        
        # Check if there's valid data
        band = dataset.GetRasterBand(1)
        stats = band.GetStatistics(True, True)
        print(f"Band statistics - Min: {stats[0]}, Max: {stats[1]}, Mean: {stats[2]}, StdDev: {stats[3]}")
        
        # Close the dataset
        dataset = None
        
    except Exception as e:
        print(f"Error reading projection information: {str(e)}")

def cleanup_files():
    """
    Remove all unnecessary files, keeping only the TIFF files.
    """
    print("\nCleaning up temporary files...")
    
    # Remove all HDF files
    hdf_files = glob.glob('./downloads/*.hdf')
    for file in hdf_files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {str(e)}")
    
    # Remove all .aux.xml files
    aux_files = glob.glob('./downloads/*.aux.xml')
    for file in aux_files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {str(e)}")
    
    # Remove any other non-TIFF files that might be generated
    all_files = glob.glob('./downloads/*')
    for file in all_files:
        if not file.lower().endswith('.tiff') and not file.lower().endswith('.tif'):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"Removed: {file}")
            except Exception as e:
                print(f"Failed to remove {file}: {str(e)}")
    
    print("Cleanup complete! Only TIFF files remain.")

if __name__ == "__main__":
    # Verify credentials first
    auth = verify_credentials()
    
    # Prince of Wales Island centered bounding box
    aoi = {
        'min_lon': -100.0,  # Western boundary
        'min_lat': 72.0,    # Southern boundary
        'max_lon': -97.0,   # Eastern boundary
        'max_lat': 73.0     # Northern boundary
    }
    
    print("\nSearching for available imagery...")
    # Use "both" to download both QKM (250m) and HKM (500m) resolution images
    available_images, product_types = get_modis_imagery(**aoi, resolution="both")
    
    # Process all available images
    if available_images:
        print(f"\nFound {len(available_images)} images. Processing all...")
        for idx, (image, product_type) in enumerate(zip(available_images, product_types)):
            print(f"\nProcessing image {idx + 1} of {len(available_images)}...")
            # Base filename without resolution - resolution will be added in the function
            output_filename = f"prince_of_wales_image_{idx + 1}.tiff"
            download_and_process_image(image, output_filename, product_type)
        
        # Clean up after all processing is complete
        cleanup_files()
        
        # Report on remaining files
        tiff_files = glob.glob('./downloads/*.tiff') + glob.glob('./downloads/*.tif')
        print(f"\nProcessing complete. {len(tiff_files)} TIFF files remain:")
        for tiff in tiff_files:
            print(f"- {os.path.basename(tiff)}")
    else:
        print("\nNo images available to process.")
