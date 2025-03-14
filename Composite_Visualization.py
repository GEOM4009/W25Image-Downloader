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

def get_modis_imagery(min_lon, min_lat, max_lon, max_lat, hours_ago=3, resolution="both"):
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
    return hdf_file, resolution_output_filename

def extract_band_from_hdf(hdf_file, band_name, band_index, output_file):
    """
    Extract a specific band from an HDF file
    
    Parameters:
    - hdf_file: Path to the HDF file
    - band_name: Name of the subdataset (e.g., "EV_250_RefSB" or "EV_500_RefSB")
    - band_index: The 0-based index of the band within the subdataset
    - output_file: Path to save the extracted band
    
    Returns:
    - Path to the output file or None if failed
    """
    try:
        # Open the HDF file
        hdf_dataset = gdal.Open(hdf_file)
        
        if hdf_dataset is None:
            print(f"Failed to open HDF file: {hdf_file}")
            return None
        
        # Find the subdataset
        subdatasets = hdf_dataset.GetSubDatasets()
        subdataset_path = None
        
        for subdataset in subdatasets:
            if band_name in subdataset[0]:
                subdataset_path = subdataset[0]
                break
        
        if subdataset_path is None:
            print(f"Subdataset {band_name} not found in {hdf_file}")
            return None
        
        # Open the subdataset
        subdataset = gdal.Open(subdataset_path)
        
        # Get the band
        band_array = subdataset.GetRasterBand(band_index + 1).ReadAsArray()
        
        # Create a dataset for the band
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(
            output_file, 
            subdataset.RasterXSize, 
            subdataset.RasterYSize, 
            1, 
            gdal.GDT_Float32
        )
        
        # Set geotransform and projection from subdataset if available
        if subdataset.GetGeoTransform():
            out_dataset.SetGeoTransform(subdataset.GetGeoTransform())
        
        if subdataset.GetProjection():
            out_dataset.SetProjection(subdataset.GetProjection())
        
        # Write the band data
        out_dataset.GetRasterBand(1).WriteArray(band_array)
        
        # Close datasets
        out_dataset = None
        subdataset = None
        hdf_dataset = None
        
        return output_file
        
    except Exception as e:
        print(f"Error extracting band: {str(e)}")
        return None




#new part 

def create_composite(qkm_file, hkm_file, output_filename):
    """
    Create a composite image using bands from QKM and HKM MODIS files.
    Returns the composite dataset path if successful, else None.
    """
    try:
        print("\nCreating composite image...")

        # Temporary directory for intermediate files
        temp_dir = "./downloads/temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Locate processed QKM and HKM files
        qkm_processed = os.path.join("./downloads", qkm_file)
        hkm_processed = os.path.join("./downloads", hkm_file)

        print(f"Using QKM file: {qkm_processed}")
        print(f"Using HKM file: {hkm_processed}")

        # Open QKM reference file
        reference_ds = gdal.Open(qkm_processed)
        if reference_ds is None:
            print(f"Failed to open reference file: {qkm_processed}")
            return None

        # Get projection and geotransform
        target_proj = reference_ds.GetProjection()
        target_geotrans = reference_ds.GetGeoTransform()
        target_width, target_height = reference_ds.RasterXSize, reference_ds.RasterYSize
        reference_ds = None

        # Define output path
        composite_path = os.path.join("./downloads", output_filename)

        # Resample HKM to match QKM resolution
        temp_hkm_resampled = os.path.join(temp_dir, "hkm_resampled.tif")
        warp_options = gdal.WarpOptions(
            format='GTiff',
            width=target_width,
            height=target_height,
            dstSRS=target_proj,
            outputBounds=(
                target_geotrans[0],
                target_geotrans[3] + target_height * target_geotrans[5],
                target_geotrans[0] + target_width * target_geotrans[1],
                target_geotrans[3]
            ),
            outputType=gdal.GDT_Float32,
            resampleAlg=gdal.GRA_Bilinear,
            multithread=True,
            dstNodata=np.nan
        )

        print("Resampling 500m data to match 250m resolution...")
        gdal.Warp(temp_hkm_resampled, hkm_processed, options=warp_options)

        return (qkm_processed, temp_hkm_resampled, composite_path, target_proj, target_geotrans, target_width, target_height)
    
    except Exception as e:
        print(f"Error in create_composite: {e}")
        return None


def apply_rgb(qkm_processed, hkm_processed, composite_path, target_proj, target_geotrans, target_width, target_height):
    """
    Reads and applies RGB bands to create a true-color composite.
    """
    try:
        print("Applying RGB bands...")

        # Create output dataset with an alpha channel
        driver = gdal.GetDriverByName("GTiff")
        composite_ds = driver.Create(
            composite_path, target_width, target_height, 4, gdal.GDT_Byte,
            options=['ALPHA=YES', 'COMPRESS=LZW']
        )

        composite_ds.SetProjection(target_proj)
        composite_ds.SetGeoTransform(target_geotrans)

        # Read QKM (Red band)
        qkm_ds = gdal.Open(qkm_processed)
        if qkm_ds is None:
            print(f"Failed to open QKM file: {qkm_processed}")
            return

        if qkm_ds.RasterCount >= 1:
            red_array = qkm_ds.GetRasterBand(1).ReadAsArray()
        else:
            print("QKM file doesn't have enough bands")
            return

        qkm_ds = None

        # Read HKM (Green & Blue bands)
        hkm_ds = gdal.Open(hkm_processed)
        if hkm_ds is None:
            print(f"Failed to open HKM file: {hkm_processed}")
            return

        if hkm_ds.RasterCount >= 2:
            green_array = hkm_ds.GetRasterBand(2).ReadAsArray()  # Band 4 (green)
            blue_array = hkm_ds.GetRasterBand(1).ReadAsArray()   # Band 3 (blue)
        else:
            print("HKM file doesn't have enough bands")
            return

        hkm_ds = None

        # Stack RGB bands into output dataset
        composite_ds.GetRasterBand(1).WriteArray(red_array)
        composite_ds.GetRasterBand(2).WriteArray(green_array)
        composite_ds.GetRasterBand(3).WriteArray(blue_array)
        
        # Alpha channel (fully opaque)
        alpha_band = np.ones_like(red_array) * 255
        composite_ds.GetRasterBand(4).WriteArray(alpha_band)

        composite_ds = None
        print("True-color composite created successfully!")

    except Exception as e:
        print(f"Error in apply_rgb: {e}")


def apply_false_color(qkm_processed, hkm_processed, composite_path, target_proj, target_geotrans, target_width, target_height):
    """
    Reads and applies false-color bands (Blue, SWIR2, SWIR3) to create a false-color composite.
    """
    try:
        print("Applying False-Color (Blue, SWIR2, SWIR3) bands...")

        # Create output dataset with an alpha channel
        driver = gdal.GetDriverByName("GTiff")
        composite_ds = driver.Create(
            composite_path, target_width, target_height, 4, gdal.GDT_Byte,
            options=['ALPHA=YES', 'COMPRESS=LZW']
        )

        composite_ds.SetProjection(target_proj)
        composite_ds.SetGeoTransform(target_geotrans)

        # Open the processed HKM file
        hkm_ds = gdal.Open(hkm_processed)
        if hkm_ds is None:
            print(f"Failed to open HKM file: {hkm_processed}")
            return

        if hkm_ds.RasterCount >= 7:
            blue_array = hkm_ds.GetRasterBand(3).ReadAsArray()  # Band 3 (Blue)
            swir2_array = hkm_ds.GetRasterBand(6).ReadAsArray()  # Band 6 (SWIR2)
            swir3_array = hkm_ds.GetRasterBand(7).ReadAsArray()  # Band 7 (SWIR3)
        else:
            print("HKM file doesn't have enough bands for false-color")
            return

        hkm_ds = None

        # Stack False-Color bands into output dataset
        composite_ds.GetRasterBand(1).WriteArray(blue_array)  # Blue
        composite_ds.GetRasterBand(2).WriteArray(swir2_array)  # SWIR2
        composite_ds.GetRasterBand(3).WriteArray(swir3_array)  # SWIR3

        # Alpha channel (fully opaque)
        alpha_band = np.ones_like(blue_array) * 255
        composite_ds.GetRasterBand(4).WriteArray(alpha_band)

        composite_ds = None
        print("False-color composite created successfully!")

    except Exception as e:
        print(f"Error in apply_false_color: {e}")


####


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

def cleanup_files(keep_original=False):
    """
    Remove all unnecessary files, keeping only the TIFF files.
    
    Parameters:
    - keep_original: If True, keep original HDF files, otherwise remove them
    """
    print("\nCleaning up temporary files...")
    
    # Remove all HDF files if not keeping originals
    if not keep_original:
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
    if not keep_original:
        all_files = glob.glob('./downloads/*')
        for file in all_files:
            if not file.lower().endswith('.tiff') and not file.lower().endswith('.tif'):
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                        print(f"Removed: {file}")
                except Exception as e:
                    print(f"Failed to remove {file}: {str(e)}")
    
    print("Cleanup complete!")

def find_matching_granules(available_images, product_types):
    """
    Find matching QKM and HKM granules that can be used for true color composites
    
    Parameters:
    - available_images: List of DataGranule objects
    - product_types: List of product types corresponding to each granule
    
    Returns:
    - List of tuples (qkm_index, hkm_index) for matching granules
    """
    matches = []
    qkm_indices = [i for i, pt in enumerate(product_types) if "QKM" in pt]
    hkm_indices = [i for i, pt in enumerate(product_types) if "HKM" in pt]
    
    # Try to match by satellite (Terra or Aqua) and acquisition time
    for qkm_idx in qkm_indices:
        qkm_type = product_types[qkm_idx]
        satellite = "MOD" if "MOD" in qkm_type else "MYD"
        
        for hkm_idx in hkm_indices:
            hkm_type = product_types[hkm_idx]
            
            # Check if from same satellite
            if satellite in hkm_type:
                # These are potentially matching granules
                matches.append((qkm_idx, hkm_idx))
                break  # Find just one match per QKM image for now
    
    return matches

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
    
    # Find matching QKM and HKM granules for true color composites
    matching_pairs = find_matching_granules(available_images, product_types)
    
    if matching_pairs:
        print(f"\nFound {len(matching_pairs)} matching QKM/HKM granule pairs for true color composites")
        
        # Process all available images and keep track of downloaded files
        downloaded_files = {}  # Store paths to downloaded files by granule index
        
        # First, download all needed granules
        print("\nDownloading all necessary granules...")
        for i, (qkm_idx, hkm_idx) in enumerate(matching_pairs):
            # Download QKM file
            qkm_granule = available_images[qkm_idx]
            qkm_output = f"prince_of_wales_image_{i+1}_QKM.tiff"
            qkm_hdf_file, qkm_processed_file = download_and_process_image(
                qkm_granule, qkm_output, product_types[qkm_idx]
            )
            downloaded_files[qkm_idx] = qkm_hdf_file
            
            # Download HKM file
            hkm_granule = available_images[hkm_idx]
            hkm_output = f"prince_of_wales_image_{i+1}_HKM.tiff"
            hkm_hdf_file, hkm_processed_file = download_and_process_image(
                hkm_granule, hkm_output, product_types[hkm_idx]
            )
            downloaded_files[hkm_idx] = hkm_hdf_file
            
        
            # Generate composites
            true_color_output = f"prince_of_wales_truecolor_{i+1}.tiff"
            false_color_output = f"prince_of_wales_falsecolor_{i+1}.tiff"
        
            composite_data = create_composite(qkm_hdf_file, hkm_hdf_file, true_color_output)
            if composite_data:
                apply_rgb(*composite_data)  # Generate True Color Composite

            composite_data_false = create_composite(qkm_hdf_file, hkm_hdf_file, false_color_output)
            if composite_data_false:
                apply_false_color(*composite_data_false)  # Generate False Color Composite

        
        # Clean up after all processing is complete
        cleanup_files(keep_original=False)
        
        # Report on remaining files
        tiff_files = glob.glob('./downloads/*.tiff') + glob.glob('./downloads/*.tif')
        print(f"\nProcessing complete. {len(tiff_files)} TIFF files remain:")
        for tiff in tiff_files:
            print(f"- {os.path.basename(tiff)}")
    
    elif available_images:
        print("\nFound individual images but no matching QKM/HKM pairs for true color composites.")
        print("Will process individual images instead...")
        
        # Process all available images
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



