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
gdal.DontUseExceptions()


def verify_credentials():
    """
    Verify NASA Earthdata credentials and ensure successful login.
    
    This function attempts to authenticate with NASA Earthdata using stored
    credentials and performs a test search to confirm the authentication works.
    If authentication fails, it provides detailed troubleshooting steps.
    
    Returns:
        earthaccess.Auth: Authenticated earthaccess object
    
    Raises:
        SystemExit: If authentication fails
    
    Author: Leo
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
    
    This function searches for MODIS imagery from both Terra and Aqua satellites
    within the specified geographic area and time window. It supports searching for
    images at different resolutions based on the user's preference.
    
    Parameters:
        min_lon (float): Minimum longitude (western boundary) of bounding box
        min_lat (float): Minimum latitude (southern boundary) of bounding box
        max_lon (float): Maximum longitude (eastern boundary) of bounding box
        max_lat (float): Maximum latitude (northern boundary) of bounding box
        hours_ago (int): How far back in time to search for imagery, in hours
        resolution (str): "QKM" (250m), "HKM" (500m), or "both" to download both resolutions
    
    Returns:
        tuple: (all_results, result_types) where:
            - all_results: List of DataGranule objects matching the search criteria
            - result_types: List of product types corresponding to each granule
    
    Author: Leo
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
    
    This function downloads a specified MODIS granule, extracts the relevant
    subdataset, and reprojects it to an Arctic Polar Stereographic projection 
    (EPSG:3995) suitable for Arctic region analysis. The process includes 
    handling geolocation information to ensure accurate spatial representation.
    
    Parameters:
        granule (DataGranule): The DataGranule object to download
        output_filename (str): The base filename for the output file
        product_type (str): The product type (e.g., "MOD02QKM", "MYD02HKM")
    
    Returns:
        tuple: (hdf_file, resolution_output_filename) where:
            - hdf_file: Path to the downloaded HDF file
            - resolution_output_filename: Path to the processed output file
    
    Author: Leo
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
    Extract a specific band from an HDF file.
    
    This function extracts a single band from a MODIS HDF file subdataset,
    creating a standalone GeoTIFF file. The function preserves geospatial 
    information from the source dataset when available.
    
    Parameters:
        hdf_file (str): Path to the HDF file
        band_name (str): Name of the subdataset (e.g., "EV_250_RefSB" or "EV_500_RefSB")
        band_index (int): The 0-based index of the band within the subdataset
        output_file (str): Path to save the extracted band
    
    Returns:
        str or None: Path to the output file if successful, None if failed
    
    Author: Alana
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

def create_true_color_composite(qkm_file, hkm_file, output_filename):
    """
    Create a true-color composite image using bands from QKM and HKM MODIS files.
    
    This function combines 250m (red band) and 500m (green, blue bands) MODIS data
    to create a true-color composite image. The process includes resampling the
    500m bands to match the higher 250m resolution, performing appropriate scaling 
    and color balancing, and handling transparency for areas with no valid data.
    The function creates both standard and expanded versions of the composite.
    
    Parameters:
        qkm_file (str): Path to the QKM (250m) MODIS file
        hkm_file (str): Path to the HKM (500m) MODIS file
        output_filename (str): Filename for the output composite image
    
    Returns:
        str or None: Path to the composite image if successful, None if failed
    
    Author: Alana
    """
    try:
        print("\nCreating true-color composite image...")
        
        # Make temporary directory for intermediate files
        temp_dir = "./downloads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # More robust file finding logic
        qkm_basename = os.path.basename(qkm_file)
        hkm_basename = os.path.basename(hkm_file)
        
        # List all files in downloads directory
        download_files = os.listdir("./downloads")
        
        # Find files with more flexible matching
        qkm_matches = [f for f in download_files if (f.endswith("_250m.tiff") or f.endswith("_250m.tif")) and 
                       (qkm_basename in f or (f.startswith("prince_of_wales") and "_QKM" in f))]
        
        hkm_matches = [f for f in download_files if (f.endswith("_500m.tiff") or f.endswith("_500m.tif")) and 
                       (hkm_basename in f or (f.startswith("prince_of_wales") and "_HKM" in f))]
        
        # Debug output to help diagnose issues
        print(f"Looking for QKM file containing '{qkm_basename}' and ending with '_250m.tiff/tif'")
        print(f"Looking for HKM file containing '{hkm_basename}' and ending with '_500m.tiff/tif'")
        print(f"Available files in downloads directory: {download_files}")
        print(f"QKM matches found: {qkm_matches}")
        print(f"HKM matches found: {hkm_matches}")
        
        # Check if we found matching files
        if not qkm_matches:
            print("No matching QKM (250m) file found. Cannot create composite.")
            return None
            
        if not hkm_matches:
            print("No matching HKM (500m) file found. Cannot create composite.")
            return None
        
        # Use the first match found
        qkm_processed = os.path.join("./downloads", qkm_matches[0])
        hkm_processed = os.path.join("./downloads", hkm_matches[0])
        
        print(f"Using QKM file: {qkm_processed}")
        print(f"Using HKM file: {hkm_processed}")
        
        # Create a common target resolution and extent
        # Use the processed QKM file as reference for higher resolution
        reference_ds = gdal.Open(qkm_processed)
        if reference_ds is None:
            print(f"Failed to open reference file: {qkm_processed}")
            return None
            
        # Get projection and geotransform from the reference file
        target_proj = reference_ds.GetProjection()
        target_geotrans = reference_ds.GetGeoTransform()
        target_width = reference_ds.RasterXSize
        target_height = reference_ds.RasterYSize
        reference_ds = None
        
        # Define output path
        composite_path = os.path.join("./downloads", output_filename)
        
        # Create temporary resampled HKM bands at QKM resolution
        temp_hkm_resampled = os.path.join(temp_dir, "hkm_resampled.tif")
        
        # Resample HKM to match QKM resolution and extent
        # Use bilinear resampling and maintain the entire extent
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
            dstNodata=np.nan  # Use NaN for nodata to handle transparency better
        )
        
        print("Resampling 500m data to match 250m resolution...")
        gdal.Warp(temp_hkm_resampled, hkm_processed, options=warp_options)
        
        # Create output dataset for the composite with alpha channel
        driver = gdal.GetDriverByName("GTiff")
        composite_ds = driver.Create(
            composite_path, 
            target_width, 
            target_height, 
            4,  # 4 bands: R, G, B, and Alpha
            gdal.GDT_Byte,
            options=['ALPHA=YES', 'COMPRESS=LZW']  # Add alpha channel and compression
        )
        
        # Set projection and geotransform
        composite_ds.SetProjection(target_proj)
        composite_ds.SetGeoTransform(target_geotrans)
        
        # Read bands - QKM file has bands 1-2, HKM file has bands 3-5
        qkm_ds = gdal.Open(qkm_processed)
        if qkm_ds is None:
            print(f"Failed to open QKM file: {qkm_processed}")
            return None
        
        # Check number of bands
        qkm_band_count = qkm_ds.RasterCount
        print(f"QKM file has {qkm_band_count} bands")
        
        if qkm_band_count >= 1:
            red_array = qkm_ds.GetRasterBand(1).ReadAsArray()  # Band 1 (red)
        else:
            print("QKM file doesn't have enough bands")
            return None
        
        qkm_ds = None
        
        hkm_ds = gdal.Open(temp_hkm_resampled)
        if hkm_ds is None:
            print(f"Failed to open resampled HKM file: {temp_hkm_resampled}")
            return None
        
        # Check number of bands
        hkm_band_count = hkm_ds.RasterCount
        print(f"HKM file has {hkm_band_count} bands")
        
        if hkm_band_count >= 2:
            green_array = hkm_ds.GetRasterBand(2).ReadAsArray()  # Band 4 (green)
            blue_array = hkm_ds.GetRasterBand(1).ReadAsArray()   # Band 3 (blue)
        else:
            print("HKM file doesn't have enough bands")
            return None
        
        hkm_ds = None
        
        # Scale values with more robust handling
        print("Scaling and creating composite...")
        
        # Function to scale array with percentile clipping
        def scale_array(array, low_clip=2, high_clip=98):
            # Make a copy to avoid modifying the original
            array_copy = array.copy()
            
            # Create mask for valid data
            valid_mask = ~np.isnan(array_copy) & (array_copy > 0)
            
            if np.sum(valid_mask) > 0:
                # Get percentiles from valid data only
                low_val = np.percentile(array_copy[valid_mask], low_clip)
                high_val = np.percentile(array_copy[valid_mask], high_clip)
                
                # Apply scaling only to valid data
                array_copy[valid_mask] = np.clip((array_copy[valid_mask] - low_val) / 
                                              (high_val - low_val) * 255, 0, 255)
                
                # Set invalid data to 0
                array_copy[~valid_mask] = 0
                
            else:
                # If no valid data, return zeros
                array_copy = np.zeros_like(array_copy)
                
            return array_copy.astype(np.uint8)
        
        # Scale each band
        red_scaled = scale_array(red_array)
        green_scaled = scale_array(green_array)
        blue_scaled = scale_array(blue_array)
        
        # Create alpha channel based on valid data in all three bands
        # Pixels with 0 in any band will be transparent
        alpha_channel = np.ones_like(red_scaled) * 255  # Start with full opacity
        
        # Create combined valid data mask
        valid_data_mask = (red_array > 0) & (green_array > 0) & (blue_array > 0)
        valid_data_mask = valid_data_mask & ~np.isnan(red_array) & ~np.isnan(green_array) & ~np.isnan(blue_array)
        
        # Set alpha to 0 (transparent) for invalid data
        alpha_channel[~valid_data_mask] = 0
        
        # Write bands to composite
        composite_ds.GetRasterBand(1).WriteArray(red_scaled)
        composite_ds.GetRasterBand(2).WriteArray(green_scaled)
        composite_ds.GetRasterBand(3).WriteArray(blue_scaled)
        composite_ds.GetRasterBand(4).WriteArray(alpha_channel)  # Alpha channel
        
        # Set color interpretation
        composite_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        composite_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        composite_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        composite_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
        # Set nodata value
        composite_ds.GetRasterBand(1).SetNoDataValue(0)
        composite_ds.GetRasterBand(2).SetNoDataValue(0)
        composite_ds.GetRasterBand(3).SetNoDataValue(0)
        
        # Close dataset
        composite_ds = None
        
        # Create another version with expanded extent if needed
        # This ensures we get the full area without clipping
        expanded_composite_path = os.path.join("./downloads", "expanded_" + output_filename)
        
        print("Creating expanded composite with full extent...")
        gdal.Warp(expanded_composite_path, composite_path, 
          format='GTiff',
          srcNodata=0,
          dstNodata=0,
          multithread=True,
          resampleAlg=gdal.GRA_NearestNeighbour,
          creationOptions=['ALPHA=YES', 'COMPRESS=LZW'])
        
        print(f"True-color composite saved to: {composite_path}")
        print(f"Expanded true-color composite saved to: {expanded_composite_path}")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return composite_path
        
    except Exception as e:
        print(f"Error creating true-color composite: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_projection(filename):
    """
    Verify the projection of a processed image.
    
    This function opens a processed GeoTIFF file and inspects its projection
    information, geotransform, dimensions, and statistics to verify
    that the reprojection was successful and the data is valid.
    
    Parameters:
        filename (str): Path to the GeoTIFF file to verify
    
    Author: Zack
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
    
    This function cleans up temporary and intermediate files created during 
    processing. By default, it removes all HDF files and auxiliary files,
    keeping only the final TIFF products. This helps manage disk space
    especially when processing multiple images.
    
    Parameters:
        keep_original (bool): If True, keep original HDF files, otherwise remove them
    
    Author: Zack
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
    Find matching QKM and HKM granules that can be used for true color composites.
    
    This function pairs 250m (QKM) and 500m (HKM) granules from the same satellite
    (Terra or Aqua) that can be combined to create true color composite images.
    The matching is based on the product type prefixes ("MOD" for Terra and "MYD" 
    for Aqua) to ensure the granules are from the same satellite pass.
    
    Parameters:
        available_images (list): List of DataGranule objects
        product_types (list): List of product types corresponding to each granule
    
    Returns:
        list: List of tuples (qkm_index, hkm_index) for matching granules that
              can be used to create true color composites
    
    Author: Alana
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
    """
    Main execution block for Arctic Image Downloader.
    
    This script downloads and processes MODIS imagery for the Arctic region,
    specifically focusing on Prince of Wales Island. It handles downloading both
    250m (QKM) and 500m (HKM) resolution imagery, processes them into the
    Arctic Polar Stereographic projection, and creates true color composites
    when matching image pairs are available.
    
    Author: Zack
    """
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
            
            # Create true color composite
            composite_output = f"prince_of_wales_truecolor_{i+1}.tiff"
            create_true_color_composite(qkm_hdf_file, hkm_hdf_file, composite_output)
        
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
