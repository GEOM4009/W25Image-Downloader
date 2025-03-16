import subprocess
import sys

def check_and_install_dependencies():
    """
    Check if required libraries are installed and install them if missing.
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
    else:
        print("\nAll dependencies are installed successfully!")
