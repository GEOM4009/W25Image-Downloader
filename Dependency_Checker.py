import subprocess
import sys
import importlib
import os

def check_and_install_dependencies():
    """
    Check if required libraries are installed and install them if missing.
    Also checks for HDF4 support in GDAL.
    Returns a tuple of (successfully_installed, failed_installations)
    """
    required_packages = [
        'earthdata',
        'rasterio',
        'geopandas',
        'requests',
        'shapely',
        # Additional dependencies for Arctic_Image_Downloader.py
        'earthaccess',
        'numpy',
        'gdal',
        'osgeo'
    ]
    
    installed = []
    failed = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
            installed.append(package)
        except ImportError:
            print(f"✗ {package} is not installed. Attempting to install...")
            try:
                # Special case for osgeo which is part of GDAL
                if package == 'osgeo':
                    if 'gdal' in installed:
                        print(f"✓ {package} is available via GDAL installation")
                        installed.append(package)
                        continue
                    else:
                        print(f"✗ {package} requires GDAL. Skipping explicit installation...")
                        continue
                
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
                installed.append(package)
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                failed.append(package)
    
    # Check for HDF4 support in GDAL
    if 'gdal' in installed:
        try:
            from osgeo import gdal
            drivers = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
            if 'HDF4' in drivers:
                print("✓ GDAL HDF4 driver is available")
            else:
                print("✗ GDAL HDF4 driver is not available. Installing via conda...")
                try:
                    # Check if we're in a conda environment
                    if os.environ.get('CONDA_DEFAULT_ENV') is not None:
                        subprocess.check_call(['conda', 'install', '-c', 'conda-forge', 'libgdal-hdf4', '-y'])
                        print("✓ Successfully installed libgdal-hdf4")
                    else:
                        print("✗ HDF4 driver needs to be installed using conda. Please run: conda install -c conda-forge libgdal-hdf4")
                        failed.append('libgdal-hdf4')
                except subprocess.CalledProcessError:
                    print("✗ Failed to install libgdal-hdf4")
                    print("Please manually run: conda install -c conda-forge libgdal-hdf4")
                    failed.append('libgdal-hdf4')
        except Exception as e:
            print(f"✗ Error checking HDF4 support: {str(e)}")
            print("Please manually run: conda install -c conda-forge libgdal-hdf4")
            failed.append('libgdal-hdf4')
    
    return installed, failed

if __name__ == "__main__":
    print("Checking and installing required dependencies...")
    installed, failed = check_and_install_dependencies()
    
    if failed:
        print("\nFailed to install the following packages:")
        for package in failed:
            print(f"- {package}")
        print("\nPlease try installing them manually using:")
        print("pip install package_name")
        
        # GDAL installation hint
        if 'gdal' in failed or 'osgeo' in failed:
            print("\nNote: GDAL can be difficult to install with pip.")
            print("On Windows, try: pip install GDAL-binaries")
            print("On macOS, try: brew install gdal && pip install gdal")
            print("On Linux, try: sudo apt-get install libgdal-dev && pip install gdal==<matching-version>")
        
        # HDF4 hint
        if 'libgdal-hdf4' in failed:
            print("\nFor HDF4 support in GDAL, run:")
            print("conda install -c conda-forge libgdal-hdf4")
    else:
        print("\nAll dependencies are installed successfully!")
