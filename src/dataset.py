"""
This module defines the WeatherSubset dataset class for loading and processing weather data
from NetCDF files. It supports extracting atmospheric and surface variables while applying
preprocessing, transformations, and normalization.

Classes:
- WeatherSubset: A PyTorch dataset that loads pairs of consecutive timesteps from NetCDF files.

Dependencies:
- torch
- torch.utils.data
- xarray
- numpy
- config
"""

import torch
import torch.utils.data as data
import xarray as xr
import numpy as np
from config import config

class WeatherSubset(data.Dataset):
    """
    A PyTorch dataset class that loads pairs of consecutive timesteps (current and future) from NetCDF files.
    
    Args:
        data_path (str): Path to the NetCDF dataset.
        max_mb (int): Maximum memory (in megabytes) to allocate for loading data.
    
    Returns:
        Tuple:
            (current_atmos, current_surface, lead_time): Current weather conditions.
            (target_atmos, target_surface): Target future weather conditions.
    """
    def __init__(self, data_path: str, max_mb: int = 500) -> None:
        self.ds: xr.Dataset = xr.open_dataset(data_path)
        self.max_lead_hours: int = 24
        self.time_resolution: int = 6
        
        needed_vars = config['surface_vars'] + config['atmos_vars']
        self.vars: list = [v for v in self.ds.data_vars if 'time' in self.ds[v].dims and v in needed_vars]

        bytes_per_pair: int = sum(
            self.ds[var].isel(time=0).nbytes for var in self.vars
        ) * 2
        
        self.max_timesteps: int = int((max_mb * 1e6) / bytes_per_pair)
        self.max_timesteps = min(self.max_timesteps, len(self.ds.time) - 4)
        self.times: xr.DataArray = self.ds.time[:self.max_timesteps]
        
        print(f"Loading {len(self.times)} timestep pairs (~{bytes_per_pair * len(self.times) / 1e6:.1f}MB)")

    def __len__(self) -> int:
        """
        Returns the number of available timesteps in the dataset.
        """
        return len(self.times)

    def _process_var(self, var: str, data: xr.DataArray) -> np.ndarray:
        """
        Processes a given weather variable by handling dimensions, transformations, and normalization.
        
        Args:
            var (str): Name of the variable.
            data (xr.DataArray): The data array corresponding to the variable.
        
        Returns:
            np.ndarray: Normalized and processed variable data.
        """
        if 'level' in data.dims:
            if data.dims != ("level", "latitude", "longitude"):
                data = data.transpose("level", "latitude", "longitude")
        elif data.dims == ("longitude", "latitude"):
            data = data.transpose("latitude", "longitude")
        
        arr: np.ndarray = data.values
        
        if var in ["2m_temperature", "temperature"]:
            arr = arr - 273.15  # Convert Kelvin to Celsius
        elif var == "10m_wind_speed":
            arr = np.log(arr + 0.1)  # Apply log transformation to wind speed
        
        stats = config["normalization_stats"][var]
        arr = (arr - stats["mean"]) / stats["std"]  # Normalize using stored stats
        return arr

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample at the given index, including both current and target timestep pairs.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple:
                (current_atmos, current_surface, lead_time): Current weather data tensors.
                (target_atmos, target_surface): Target weather data tensors.
        """
        Δt_steps: int = torch.randint(1, 5, (1,)).item()
        target_idx: int = idx + Δt_steps
        
        current_atmos, current_surface, target_atmos, target_surface = [], [], [], []

        for var in config['atmos_vars']:
            current_data = self.ds[var].isel(time=idx)
            target_data = self.ds[var].isel(time=target_idx)
            current_atmos.append(torch.tensor(self._process_var(var, current_data)))
            target_atmos.append(torch.tensor(self._process_var(var, target_data)))

        for var in config['surface_vars']:
            current_data = self.ds[var].isel(time=idx)
            target_data = self.ds[var].isel(time=target_idx)
            current_surface.append(torch.tensor(self._process_var(var, current_data)).unsqueeze(0))
            target_surface.append(torch.tensor(self._process_var(var, target_data)).unsqueeze(0))

        current_atmos_tensor = torch.cat(current_atmos, dim=0)
        current_surface_tensor = torch.cat(current_surface, dim=0)
        target_atmos_tensor = torch.cat(target_atmos, dim=0)
        target_surface_tensor = torch.cat(target_surface, dim=0)
        
        lead_time: torch.Tensor = torch.tensor([Δt_steps * self.time_resolution / self.max_lead_hours])
        
        return (current_atmos_tensor, current_surface_tensor, lead_time), (target_atmos_tensor, target_surface_tensor)
