# Download the dataset in the way required by the program to run

import xarray as xr
import fsspec
from datetime import datetime

# Configuration
START_DATE = "2016-01-01"
END_DATE = "2016-02-01"
OUTPUT_FILE = "weather_data_jan2016.nc"

# Create filesystem connection
fs = fsspec.filesystem('gcs', anon=True)
store = fs.get_mapper('weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr')

# Open dataset and select time range
with xr.open_zarr(store, consolidated=True) as ds:
    # Select 1 month of data
    monthly_data = ds.sel(time=slice(START_DATE, END_DATE))
    
    # Set encoding for compression
    encoding = {
        var: {'zlib': True, 'complevel': 5} 
        for var in monthly_data.data_vars
    }
    
    # Save to NetCDF
    monthly_data.load().to_netcdf(
        OUTPUT_FILE,
        encoding=encoding,
        mode='w'
    )

print(f"Saved {START_DATE} to {END_DATE} data to {OUTPUT_FILE}")
print(f"File size: {os.path.getsize(OUTPUT_FILE)/1e6:.1f} MB")
