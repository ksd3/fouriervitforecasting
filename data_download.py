import xarray as xr
import fsspec
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Access the zarr store directly from Google Cloud Storage
gcs_path = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"
mapper = fsspec.get_mapper(gcs_path, anon=True)
ds = xr.open_zarr(mapper)

# Basic dataset information
print("Dataset Overview:")
print(f"Dimensions: {ds.dims}")
print(f"Coordinates: {list(ds.coords)}")
print(f"Data variables: {list(ds.data_vars)}")
print("\n")

# Time information
print("Time Information:")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Time steps: {len(ds.time)}")
print(f"Time resolution: {ds.time.values[1] - ds.time.values[0]}")
print("\n")

# Spatial information
print("Spatial Information:")
print(f"Longitude range: {ds.longitude.values.min()} to {ds.longitude.values.max()}")
print(f"Latitude range: {ds.latitude.values.min()} to {ds.latitude.values.max()}")
print(f"Grid resolution: {len(ds.longitude)}x{len(ds.latitude)}")
print("\n")

# If there are pressure levels, examine them
if 'level' in ds.dims:
    print("Pressure Levels:")
    print(f"Levels: {ds.level.values}")
    print(f"Number of levels: {len(ds.level)}")
    print("\n")

# Examine each variable in more detail
print("Variable Details:")
for var_name in ds.data_vars:
    var = ds[var_name]
    print(f"Variable: {var_name}")
    print(f"  Dimensions: {var.dims}")
    print(f"  Shape: {var.shape}")
    if hasattr(var, 'units'):
        print(f"  Units: {var.units}")
    if hasattr(var, 'long_name'):
        print(f"  Long name: {var.long_name}")
    print(f"  Data type: {var.dtype}")
    
    # Try to get min/max values from a single time step to avoid loading too much data
    try:
        sample = var.isel(time=0)
        if 'level' in var.dims:
            sample = sample.isel(level=0)
        print(f"  Sample min/max (first time step): {float(sample.min().values):.4f} to {float(sample.max().values):.4f}")
    except:
        print("  Could not compute sample min/max")
    print()

# Categorize variables by type
surface_vars = [var for var in ds.data_vars if 'level' not in ds[var].dims]
atmospheric_vars = [var for var in ds.data_vars if 'level' in ds[var].dims]

print(f"Surface variables ({len(surface_vars)}): {surface_vars}")
print(f"Atmospheric variables ({len(atmospheric_vars)}): {atmospheric_vars}")

# Visualize a sample of one surface and one atmospheric variable
if surface_vars:
    var_name = surface_vars[0]
    plt.figure(figsize=(12, 6))
    ds[var_name].isel(time=0).plot()
    plt.title(f"Sample Surface Variable: {var_name}")
    plt.show()

if atmospheric_vars:
    var_name = atmospheric_vars[0]
    plt.figure(figsize=(12, 6))
    ds[var_name].isel(time=0, level=0).plot()
    plt.title(f"Sample Atmospheric Variable: {var_name} (Level: {ds.level.values[0]})")
    plt.show()
