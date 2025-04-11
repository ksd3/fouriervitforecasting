import xarray as xr
import fsspec
import numpy as np
from scipy import stats
import os
import gc

# Configuration
START_DATE = "2016-01-01"
END_DATE = "2016-02-01"
VARIABLES = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed',
    '2m_temperature', 'geopotential', 'mean_sea_level_pressure',
    'specific_humidity', 'surface_pressure', 'temperature',
    'total_precipitation_6hr', 'u_component_of_wind', 'v_component_of_wind',
    'vertical_velocity', 'wind_speed'
]
OUTPUT_FILE = "variable_analysis_report.txt"


def calculate_storage(var):
    """Calculate storage requirements"""
    element_size = 4
    elements = np.prod(var.shape)
    return (elements * element_size) / 1e6


def analyze_variable(var_name):
    """Complete analysis pipeline for one variable"""
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"\n{'#'*80}\n# {var_name} Analysis\n{'#'*80}\n")

        try:
            ds = xr.open_zarr(
                fsspec.get_mapper(
                    'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr',
                    anon=True
                ), consolidated=True
            )
            
            da = ds[var_name].sel(time=slice(START_DATE, END_DATE)).load()
            storage_mb = calculate_storage(da)

            f.write(f"\n[Storage Requirements]\n")
            f.write(f"• Loaded size: {storage_mb:.1f} MB\n")
            if storage_mb > 500:
                f.write("⚠️ WARNING: Large dataset loaded into memory\n")

            f.write(f"\n[Basic Statistics]\n")
            f.write(f"• Dimensions: {da.dims}\n")
            f.write(f"• Global min: {da.min().item():.4f}\n")
            f.write(f"• Global max: {da.max().item():.4f}\n")
            f.write(f"• Mean: {da.mean().item():.4f}\n")
            f.write(f"• Std dev: {da.std().item():.4f}\n")

            f.write(f"\n[Distribution Analysis]\n")
            sample = da.values.flatten()
            sample = sample[~np.isnan(sample)]
            if np.unique(sample).size > 1:
                f.write(f"• Skewness: {stats.skew(sample):.4f}\n")
                f.write(f"• Kurtosis: {stats.kurtosis(sample):.4f}\n")
                f.write(f"• 1st percentile: {np.percentile(sample, 1):.4f}\n")
                f.write(f"• 99th percentile: {np.percentile(sample, 99):.4f}\n")
                f.write(f"• Normality test p-value: {stats.normaltest(sample).pvalue:.2e}\n")
            else:
                f.write("• Constant values detected - distribution stats skipped\n")

            zero_frac = np.mean(sample == 0)
            if zero_frac > 0.1:
                f.write(f"• Zero values: {zero_frac*100:.1f}%\n")

            f.write(f"\n[Temporal Analysis]\n")
            time_mean = da.mean(dim=['longitude', 'latitude'])
            if 'level' in da.dims:
                time_mean = time_mean.mean(dim='level')
            f.write(f"• Temporal mean range: {time_mean.min().item():.4f} to {time_mean.max().item():.4f}\n")
            f.write(f"• Temporal std: {time_mean.std().item():.4f}\n")

            del da, ds
            gc.collect()
            f.write("\nAnalysis completed successfully!\n")

        except Exception as main_err:
            f.write(f"\nERROR: {str(main_err)}\n")
        finally:
            f.write("\n" + "="*80 + "\n")



# Initialize output file
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Run analysis for all variables sequentially
for var in VARIABLES:
    analyze_variable(var)

print(f"Full analysis completed! Results saved to {OUTPUT_FILE}")
