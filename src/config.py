"""
This module defines the configuration settings for the weather forecasting project.
It includes dataset paths, batch sizes, training parameters, variable selections,
and normalization statistics for preprocessing.

Configuration Dictionary:
- data_path: Path to the NetCDF dataset.
- batch_size: Number of samples per training batch.
- patch_size: Size of the image patches used in training.
- epochs: Number of training epochs.
- max_data_mb: Maximum memory allocated for data loading.
- surface_vars: List of surface-level atmospheric variables.
- atmos_vars: List of atmospheric-level variables.
- atmos_levels: Number of vertical atmospheric levels.
- normalization_stats: Dictionary containing mean and standard deviation for each variable.

Dependencies:
- numpy
"""

import numpy as np  

config = {
    'data_path': '../weather_data_jan2016.nc',  # Path to the dataset
    'batch_size': 4,  # Number of samples per batch
    'patch_size': 64,  # Patch size for training
    'epochs': 1,  # Number of training epochs (set to 1 for demo purposes)
    'max_data_mb': 500,  # Maximum allowed data size in MB

    'surface_vars': [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '10m_wind_speed',
        '2m_temperature',
        'mean_sea_level_pressure',
        'surface_pressure',
        'total_precipitation_6hr'
    ],
    'atmos_vars': [
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'vertical_velocity',
        'wind_speed'
    ],
    'atmos_levels': 13,  # Number of atmospheric levels

    'normalization_stats': {
        "10m_u_component_of_wind": {"mean": 0.1602, "std": 5.5341},
        "10m_v_component_of_wind": {"mean": -0.2449, "std": 4.6193},
        "10m_wind_speed":          {"mean": 6.2699,  "std": 3.6444},
        "2m_temperature":          {"mean": 277.5990,"std": 19.8763},
        "geopotential":            {"mean": 77741.1953, "std": 59489.7422},
        "mean_sea_level_pressure": {"mean": 100918.0, "std": 1488.2883},
        "specific_humidity":       {"mean": 0.0017,  "std": 0.0036},
        "surface_pressure":        {"mean": 96669.8125,"std": 9505.7998},
        "temperature":             {"mean": 243.1200,"std": 28.7212},
        "total_precipitation_6hr": {"mean": 0.0,     "std": 1.0},
        "u_component_of_wind":     {"mean": 8.2444,  "std": 14.9051},
        "v_component_of_wind":     {"mean": 0.0435,  "std": 9.7611},
        "vertical_velocity":       {"mean": 0.0050,  "std": 0.1557},
        "wind_speed":              {"mean": 15.0210, "std": 12.6602}
    }
}
