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

###############################

# NASA Earthdata credentials - CHANGE THESE TO YOUR OWN VALUES
os.environ['EARTHDATA_USERNAME'] = 'username'  # Replace with your actual username
os.environ['EARTHDATA_PASSWORD'] = 'password'  # Replace with your actual password

# Area of Interest Bounding Box - CHANGE THESE TO YOUR DESIRED COORDINATES
aoi = {
    'min_lon': -76.0,  # Western boundary
    'min_lat': 45.2,   # Southern boundary
    'max_lon': -75.3,  # Eastern boundary
    'max_lat': 45.6    # Northern boundary
}

################################

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
        print("Authenticating with NASA Earthdata...")
        auth = earthaccess.login(strategy='environment')
        test_search = earthaccess.search_data(
            short_name="MOD02QKM",
            temporal=(
                datetime.datetime.now(UTC) - datetime.timedelta(hours=1),
                datetime.datetime.now(UTC)
            )
        )
        print("✓ Authentication successful")
        return auth
    except Exception as e:
        print("✗ Authentication failed:", str(e))
        print("Please check your credentials at the top of this file")
        sys.exit(1)


def get_modis_imagery(min_lon, min_lat, max_lon, max_lat, hours_ago=0.5, resolution="both"):
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
    # Calculate time range
    end_time = datetime.datetime.now(UTC)
    start_time = end_time - datetime.timedelta(hours=hours_ago)
    
    print(f"Searching for imagery (past {hours_ago} hours, {min_lat:.2f}°N/{min_lon:.2f}°W to {max_lat:.2f}°N/{max_lon:.2f}°W)")
    
    all_results = []
    result_types = []
    acquisition_times = []  # Store acquisition times
    
    products_to_search = []
    if resolution == "QKM" or resolution == "both":
        products_to_search.append(("MOD02QKM", "MYD02QKM", "250m"))
    if resolution == "HKM" or resolution == "both":
        products_to_search.append(("MOD02HKM", "MYD02HKM", "500m"))
    
    for terra_product, aqua_product, res_desc in products_to_search:
        # Search Terra MODIS
        terra_results = earthaccess.search_data(
            short_name=terra_product,
            temporal=(start_time, end_time),
            bounding_box=(min_lon, min_lat, max_lon, max_lat)
        )
        if terra_results:
            all_results.extend(terra_results)
            result_types.extend([terra_product] * len(terra_results))
            
            # Extract acquisition times for Terra results
            for result in terra_results:
                acq_time = None
                try:
                    # Try to get acquisition time from metadata
                    acq_time = result.time_start
                except:
                    # If not available, use current time as fallback
                    acq_time = datetime.datetime.now(UTC)
                acquisition_times.append(acq_time)
                
            print(f"Found {len(terra_results)} Terra {res_desc} images")
        
        # Search Aqua MODIS
        aqua_results = earthaccess.search_data(
            short_name=aqua_product,
            temporal=(start_time, end_time),
            bounding_box=(min_lon, min_lat, max_lon, max_lat)
        )
        if aqua_results:
            all_results.extend(aqua_results)
            result_types.extend([aqua_product] * len(aqua_results))
            
            # Extract acquisition times for Aqua results
            for result in aqua_results:
                acq_time = None
                try:
                    # Try to get acquisition time from metadata
                    acq_time = result.time_start
                except:
                    # If not available, use current time as fallback
                    acq_time = datetime.datetime.now(UTC)
                acquisition_times.append(acq_time)
                
            print(f"Found {len(aqua_results)} Aqua {res_desc} images")
    
    return all_results, result_types, acquisition_times


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
    
    resolution = "unknown"
    if "QKM" in product_type:
        resolution = "250m"
    elif "HKM" in product_type:
        resolution = "500m"
    
    base_name, ext = os.path.splitext(output_filename)
    resolution_output_filename = f"{base_name}_{resolution}{ext}"
    
    print(f"Downloading and processing {product_type} ({resolution})...")
    downloaded_files = earthaccess.download(granule, local_path="./downloads")
    hdf_file = None
    
    if downloaded_files:
        hdf_file = downloaded_files[0]
        
        gdal.AllRegister()
        hdf_dataset = gdal.Open(hdf_file)
        
        if hdf_dataset is None:
            print(f"Failed to open HDF file: {hdf_file}")
            return
        
        subdatasets = hdf_dataset.GetSubDatasets()
        subdataset_path = subdatasets[0][0]
        
        temp_output = os.path.join("./downloads", "temp_" + resolution_output_filename)
        final_output = os.path.join("./downloads", resolution_output_filename)
        
        try:
            subdataset = gdal.Open(subdataset_path)
            if subdataset is None:
                print("Failed to open subdataset")
                return
            
            geoloc_dataset = gdal.Open(hdf_file)
            latitude = geoloc_dataset.GetSubDatasets()[0][0]
            longitude = geoloc_dataset.GetSubDatasets()[1][0]
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3995)  # Arctic Polar Stereographic
            
            warp_options = gdal.WarpOptions(
                format='GTiff',
                dstSRS=srs.ExportToWkt(),
                outputType=gdal.GDT_Float32,
                resampleAlg=gdal.GRA_Bilinear,
                geoloc=True,
                multithread=True,
                warpOptions=['NUM_THREADS=ALL_CPUS']
            )
            
            gdal.Warp(final_output, subdataset, options=warp_options)
            
            # Get basic stats
            dataset = gdal.Open(final_output)
            if dataset:
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                print(f"Processed image: {width}x{height} ({os.path.basename(final_output)})")
                dataset = None
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
        
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            hdf_dataset = None
            if 'subdataset' in locals() and subdataset:
                subdataset = None
            if 'geoloc_dataset' in locals() and geoloc_dataset:
                geoloc_dataset = None
            
    else:
        print("Failed to download image.")
    
    return hdf_file, resolution_output_filename


def cleanup_files(keep_intermediate=False):
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
    print("Cleaning up temporary files...")
    
    # Remove all HDF files
    for file in glob.glob('./downloads/*.hdf'):
        try:
            os.remove(file)
        except Exception:
            pass
    
    # Remove all .aux.xml files
    for file in glob.glob('./downloads/*.aux.xml'):
        try:
            os.remove(file)
        except Exception:
            pass
    
    # Remove intermediate QKM and HKM files unless explicitly kept
    if not keep_intermediate:
        for file in glob.glob('./downloads/*_QKM_*.tiff') + glob.glob('./downloads/*_QKM_*.tif'):
            try:
                os.remove(file)
                print(f"Removed intermediate file: {os.path.basename(file)}")
            except Exception:
                pass
                
        for file in glob.glob('./downloads/*_HKM_*.tiff') + glob.glob('./downloads/*_HKM_*.tif'):
            try:
                os.remove(file)
                print(f"Removed intermediate file: {os.path.basename(file)}")
            except Exception:
                pass
    
    # Remove any other non-TIFF files
    for file in glob.glob('./downloads/*'):
        if not file.lower().endswith('.tiff') and not file.lower().endswith('.tif'):
            try:
                if os.path.isfile(file):
                    os.remove(file)
            except Exception:
                pass


def create_multi_band_composite(qkm_file, hkm_file, output_filename, acquisition_time=None):
    """
    Create a multi band composite image using bands from QKM and HKM MODIS files.
    
    This function combines 250m and 500m MODIS data to create a multi band composite
    image. The process includes resampling the 500m bands to match the higher 250m resolution, 
    performing appropriate scaling and color balancing, and handling transparency for areas 
    with no valid data. 
    
    Parameters:
        qkm_file (str): Path to the QKM (250m) MODIS file
        hkm_file (str): Path to the HKM (500m) MODIS file
        output_filename (str): Filename for the output composite image
    
    Returns:
        str or None: Path to the composite image if successful, None if failed
    
    Author: Alana
    """
    try:
        # Generate date-time based filename if acquisition time is provided
        if acquisition_time:
            # Format: Composite_YYYYMMDD_HHMMSS.tiff
            date_str = acquisition_time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"Composite_{date_str}.tiff"
            print(f"Creating composite image for {date_str}...")
        else:
            print(f"Creating composite image...")
        
        temp_dir = "./downloads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        download_files = os.listdir("./downloads")
        
        qkm_matches = [f for f in download_files if (f.endswith("_250m.tiff") or f.endswith("_250m.tif")) and 
                       (os.path.basename(qkm_file) in f or "_QKM" in f)]
        
        hkm_matches = [f for f in download_files if (f.endswith("_500m.tiff") or f.endswith("_500m.tif")) and 
                       (os.path.basename(hkm_file) in f or "_HKM" in f)]
        
        if not qkm_matches or not hkm_matches:
            print("No matching QKM or HKM files found.")
            return None
        
        qkm_processed = os.path.join("./downloads", qkm_matches[0])
        hkm_processed = os.path.join("./downloads", hkm_matches[0])
        
        qkm_ds = gdal.Open(qkm_processed)
        hkm_ds = gdal.Open(hkm_processed)
        
        if qkm_ds is None or hkm_ds is None:
            print("Failed to open source datasets")
            return None
        
        target_proj = qkm_ds.GetProjection()
        target_geotrans = qkm_ds.GetGeoTransform()
        target_width = qkm_ds.RasterXSize
        target_height = qkm_ds.RasterYSize
        
        qkm_band_count = qkm_ds.RasterCount
        hkm_band_count = hkm_ds.RasterCount
        
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
        
        gdal.Warp(temp_hkm_resampled, hkm_processed, options=warp_options)
        
        hkm_resampled_ds = gdal.Open(temp_hkm_resampled)
        
        total_bands = qkm_band_count + hkm_band_count
        
        composite_path = os.path.join("./downloads", output_filename)
        driver = gdal.GetDriverByName("GTiff")
        
        # Add BIGTIFF=YES option to handle large files
        composite_ds = driver.Create(
            composite_path, 
            target_width, 
            target_height, 
            total_bands,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'BIGTIFF=YES']
        )
        
        composite_ds.SetProjection(target_proj)
        composite_ds.SetGeoTransform(target_geotrans)
        
        def scale_array(array):
            array_copy = array.copy()
            valid_mask = ~np.isnan(array_copy) & (array_copy > 0)
            
            if np.sum(valid_mask) > 0:
                low_val = np.percentile(array_copy[valid_mask], 2)
                high_val = np.percentile(array_copy[valid_mask], 98)
                
                array_copy[valid_mask] = np.clip(
                    (array_copy[valid_mask] - low_val) / (high_val - low_val), 
                    0, 1
                )
                
                array_copy[~valid_mask] = 0
            else:
                array_copy = np.zeros_like(array_copy)
            
            return array_copy
        
        # Copy QKM bands
        for i in range(qkm_band_count):
            band_array = qkm_ds.GetRasterBand(i+1).ReadAsArray()
            scaled_array = scale_array(band_array)
            composite_ds.GetRasterBand(i+1).WriteArray(scaled_array)
            composite_ds.GetRasterBand(i+1).SetDescription(f"QKM Band {i+1}")
        
        # Copy HKM bands
        for i in range(hkm_band_count):
            band_array = hkm_resampled_ds.GetRasterBand(i+1).ReadAsArray()
            scaled_array = scale_array(band_array)
            composite_ds.GetRasterBand(qkm_band_count + i+1).WriteArray(scaled_array)
            composite_ds.GetRasterBand(qkm_band_count + i+1).SetDescription(f"HKM Band {i+1}")
        
        # Close all datasets
        composite_ds = None
        qkm_ds = None
        hkm_ds = None
        hkm_resampled_ds = None
        
        print(f"Created composite image with {total_bands} bands: {os.path.basename(composite_path)}")
        
        # Clean up temporary files
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
        print(f"Error creating composite: {str(e)}")
        return None


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
    auth = verify_credentials()
    
    print("Searching for MODIS imagery...")
    available_images, product_types, acquisition_times = get_modis_imagery(**aoi, resolution="both")
    
    if not available_images:
        print("No images available to process.")
        sys.exit(0)
    
    # Find matching QKM and HKM granules for composites
    matching_pairs = []
    
    qkm_indices = [i for i, pt in enumerate(product_types) if "QKM" in pt]
    hkm_indices = [i for i, pt in enumerate(product_types) if "HKM" in pt]
    
    # Match by satellite (Terra or Aqua)
    for qkm_idx in qkm_indices:
        qkm_type = product_types[qkm_idx]
        satellite = "MOD" if "MOD" in qkm_type else "MYD"
        
        for hkm_idx in hkm_indices:
            hkm_type = product_types[hkm_idx]
            if satellite in hkm_type:
                matching_pairs.append((qkm_idx, hkm_idx))
                break
    
    if matching_pairs:
        print(f"Found {len(matching_pairs)} matching QKM/HKM pairs for composites")
        
        downloaded_files = {}
        
        for i, (qkm_idx, hkm_idx) in enumerate(matching_pairs):
            # Get acquisition time for naming
            acq_time = acquisition_times[qkm_idx]
            
            # Download QKM file
            qkm_granule = available_images[qkm_idx]
            qkm_output = f"temp_image_{i+1}_QKM.tiff"
            qkm_hdf_file, qkm_processed_file = download_and_process_image(
                qkm_granule, qkm_output, product_types[qkm_idx]
            )
            downloaded_files[qkm_idx] = qkm_hdf_file
            
            # Download HKM file
            hkm_granule = available_images[hkm_idx]
            hkm_output = f"temp_image_{i+1}_HKM.tiff"
            hkm_hdf_file, hkm_processed_file = download_and_process_image(
                hkm_granule, hkm_output, product_types[hkm_idx]
            )
            downloaded_files[hkm_idx] = hkm_hdf_file
            
            # Create multi-band composite with datetime-based filename
            create_multi_band_composite(
                qkm_processed_file, 
                hkm_processed_file, 
                f"Composite_{i+1}.tiff",  # This will be replaced by the datetime version
                acquisition_time=acq_time
            )
        
        # Clean up after all processing is complete - delete intermediate files too
        cleanup_files(keep_intermediate=False)
        
        # Report on remaining files
        tiff_files = glob.glob('./downloads/*.tiff') + glob.glob('./downloads/*.tif')
        print(f"Processing complete. {len(tiff_files)} TIFF files available in ./downloads/")
    
    elif available_images:
        print("Processing individual images...")
        
        for idx, (image, product_type) in enumerate(zip(available_images, product_types)):
            output_filename = f"image_{idx + 1}.tiff"
            download_and_process_image(image, output_filename, product_type)
        
        cleanup_files()
        
        tiff_files = glob.glob('./downloads/*.tiff') + glob.glob('./downloads/*.tif')
        print(f"Processing complete. {len(tiff_files)} TIFF files available in ./downloads/")
