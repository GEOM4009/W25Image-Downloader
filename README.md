# Arctic Image Downloader

A tool for downloading, processing, and creating composites of MODIS satellite imagery for Arctic regions.

## Overview

The Arctic Image Downloader allows you to search for and download recent MODIS satellite imagery from NASA's Earthdata platform, specifically focused on Arctic regions. It automatically processes the imagery into the Arctic Polar Stereographic projection (EPSG:3995) and creates multi-band composites from matching 250m and 500m resolution images.

## Features

- Search for recent MODIS imagery from both Terra and Aqua satellites
- Download imagery at 250m (QKM) and 500m (HKM) resolutions
- Automatic reprojection to Arctic Polar Stereographic (EPSG:3995)
- Create multi-band composites combining bands from both resolutions
- Automatic cleanup of intermediate files
- Time-stamped output for easy sorting and analysis

## Prerequisites

- NASA Earthdata account ([Register here](https://urs.earthdata.nasa.gov/users/new))
- Conda or Miniconda ([Installation instructions](https://docs.conda.io/en/latest/miniconda.html))

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/arctic-image-downloader.git
   cd arctic-image-downloader
   ```

2. Create the conda environment:
   ```
   conda env create -f ImageDownloaderEnvironment.yml
   ```

3. Activate the environment:
   ```
   conda activate image_downloader
   ```

## Usage

1. Edit the credentials in `Arctic_Image_Downloader.py`:
   ```python
   os.environ['EARTHDATA_USERNAME'] = 'your_username'
   os.environ['EARTHDATA_PASSWORD'] = 'your_password'
   ```

2. (Optional) Modify the area of interest (AOI) to your region:
   ```python
   aoi = {
       'min_lon': -76.0,  # Western boundary
       'min_lat': 45.2,   # Southern boundary
       'max_lon': -75.3,  # Eastern boundary
       'max_lat': 45.6    # Northern boundary
   }
   ```

3. (Optional) Adjust the time range by modifying the `hours_ago` parameter in the function call:
   ```python
   available_images, product_types, acquisition_times = get_modis_imagery(**aoi, hours_ago=5.4, resolution="both")
   ```

4. Run the downloader:
   ```
   python Arctic_Image_Downloader.py
   ```

5. The processed composites will be saved in the `./downloads` directory.

## Understanding the Output

- **Composite images**: Named `Composite_YYYYMMDD_HHMMSS.tiff` with the date and time of acquisition
- Each composite contains all bands from both the 250m and 500m imagery
- Images are in EPSG:3995 projection for optimal Arctic visualization
- All intermediate files are automatically removed
- The output is a multi-band GeoTIFF compatible with most GIS software

## Files Included

- **Arctic_Image_Downloader.py**: Main script for downloading and processing images
- **ImageDownloaderEnvironment.yml**: Conda environment file with all required dependencies

## Troubleshooting

- **Authentication Issues**: Double-check your NASA Earthdata credentials
- **No Images Found**: Try increasing the `hours_ago` parameter in the `get_modis_imagery` function call
- **GDAL Errors**: Make sure you're using the conda environment which has the proper GDAL configuration
- **File Size Errors**: The script uses BIGTIFF=YES, but extremely large areas may still cause issues

## Dependencies

This tool relies on several key libraries, all included in the Conda environment:
- earthaccess - For NASA Earthdata API access
- GDAL (with HDF4 support) - For geospatial processing
- geopandas - For geospatial data handling
- numpy - For numerical processing
- Other supporting libraries

## License

MIT License
Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- NASA for providing free access to MODIS satellite imagery
- GDAL developers for the excellent geospatial processing library
- The earthaccess library for simplified Earthdata access
- Claude.ai for all the help troubleshooting
