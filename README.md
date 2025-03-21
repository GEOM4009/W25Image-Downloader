# W25Image-Downloader

Make sure to have a file named downloads in the same directory as the code


Summary of the project
What does it do
Who's it for
How to get it going
Should be easy to understand

https://github.com/banesullivan/README

Searching for available imagery...

Search Parameters:
Time Range: From 2025-03-21 18:16:10 UTC
            To   2025-03-21 19:16:10 UTC
Area of interest: 100.00°W, 72.00°N to 97.00°W, 73.00°N
Resolution: Both 250m and 500m

Searching Terra MODIS (MOD02QKM - 250m)...
Found 1 Terra 250m images
Searching Aqua MODIS (MYD02QKM - 250m)...

Searching Terra MODIS (MOD02HKM - 500m)...
Found 1 Terra 500m images
Searching Aqua MODIS (MYD02HKM - 500m)...

Found 1 matching QKM/HKM granule pairs for true color composites

Downloading all necessary granules...
Downloading MOD02QKM granule (250m)...
QUEUEING TASKS | : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<?, ?it/s]
PROCESSING TASKS | : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.35it/s]
COLLECTING RESULTS | : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 64.02it/s]
Processing image...
C:\ProgramData\anaconda3\envs\geom\Lib\site-packages\osgeo\gdal.py:312: FutureWarning: Neither gdal.UseExceptions() nor gdal.DontUseExceptions() has been explicitly called. In GDAL 4.0, exceptions will be enabled by default.
  warnings.warn(
ERROR 4: `downloads\MOD02QKM.A2025080.1815.061.2025080191311.NRT.hdf' not recognized as being in a supported file format.
Failed to open HDF file: downloads\MOD02QKM.A2025080.1815.061.2025080191311.NRT.hdf
Traceback (most recent call last):
  File "P:\GEOM4009\FinalProject\Arctic_Image_Downloader.py", line 756, in <module>
    qkm_hdf_file, qkm_processed_file = download_and_process_image(
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable NoneType object

