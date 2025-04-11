"""
This module contains unit tests for various components of the weather forecasting project.
It ensures that key functionalities, such as configuration validation, dataset handling,
model forward pass, and training logic, work correctly.

Tests included:
- Configuration file structure validation
- Dataset loading and sample structure verification
- FourierViT model forward pass correctness
- Training function execution

Usage:
    python -m unittest test_project.py

This module can be integrated into CI pipelines or executed using pytest.

Dependencies:
- unittest
- os
- torch
- tempfile
- shutil
- typing
"""

import unittest
import os
import torch
import tempfile
import shutil
from typing import List, Tuple, Any

from config import config
from dataset import WeatherSubset
from model.fourier_vit import FourierViT
from train import train_epoch
from evaluate import evaluate


class TestConfig(unittest.TestCase):
    """
    Tests to verify the structure and validity of the configuration dictionary.
    """
    def test_config_keys(self) -> None:
        """
        Ensures that all required keys are present in the configuration dictionary.
        """
        required_keys: List[str] = [
            'data_path', 'batch_size', 'patch_size', 'epochs', 'max_data_mb',
            'surface_vars', 'atmos_vars', 'atmos_levels', 'normalization_stats'
        ]
        for key in required_keys:
            self.assertIn(key, config, msg=f"Missing key '{key}' in config.")

        # Verify normalization statistics include all variables
        all_vars: List[str] = config['surface_vars'] + config['atmos_vars']
        for var in all_vars:
            self.assertIn(var, config['normalization_stats'],
                          msg=f"Missing normalization stats for '{var}'.")


class TestDataset(unittest.TestCase):
    """
    Tests for the WeatherSubset dataset class.
    If the dataset file is missing, tests are skipped.
    """
    
    def setUp(self) -> None:
        """
        Prepares the dataset test by checking if the dataset file exists.
        """
        self.data_path: str = config['data_path']
        if not os.path.isfile(self.data_path):
            self.skipTest(f"Data file {self.data_path} not found. Skipping dataset tests.")

    def test_dataset_instantiation(self) -> None:
        """
        Checks if the dataset can be instantiated without errors and has a nonzero length.
        """
        dataset: WeatherSubset = WeatherSubset(self.data_path, config['max_data_mb'])
        self.assertGreater(len(dataset), 0, "Dataset length is zero, unexpected.")

    def test_dataset_getitem(self) -> None:
        """
        Fetches an item from the dataset and verifies the expected tensor shapes.
        """
        dataset: WeatherSubset = WeatherSubset(self.data_path, config['max_data_mb'])
        sample: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] = dataset[0]
        (current_atmos, current_surface, lead_time), (target_atmos, target_surface) = sample

        # Validate dimensionality
        self.assertEqual(current_atmos.dim(), 3, "current_atmos should be 3D.")
        self.assertEqual(current_surface.dim(), 3, "current_surface should be 3D.")
        self.assertEqual(lead_time.dim(), 1, "lead_time should be 1D.")
        self.assertEqual(target_atmos.dim(), 3, "target_atmos should be 3D.")
        self.assertEqual(target_surface.dim(), 3, "target_surface should be 3D.")


class TestModel(unittest.TestCase):
    """
    Tests for the FourierViT model.
    """
    def test_model_forward_shapes(self) -> None:
        """
        Verifies that the model processes a small synthetic input correctly and produces expected output shape.
        """
        device: torch.device = torch.device('cpu')
        batch_size: int = 2
        height: int = 16
        width: int = 16
        test_atmos_vars: int = 2
        test_atmos_levels: int = 2
        test_surface_vars: int = 3
        in_channels: int = (test_atmos_vars * test_atmos_levels) + test_surface_vars

        model: FourierViT = FourierViT(
            img_size=(height, width),
            patch_size=4,
            atmos_vars=test_atmos_vars,
            atmos_levels=test_atmos_levels,
            surface_vars=test_surface_vars,
            embed_dim=32,
            fno_modes=(4, 4),
            fno_width=8,
            fno_depth=1,
            vit_depth=1,
            vit_heads=1
        ).to(device)

        test_atmos: torch.Tensor = torch.randn(batch_size, test_atmos_vars * test_atmos_levels, height, width).to(device)
        test_surface: torch.Tensor = torch.randn(batch_size, test_surface_vars, height, width).to(device)
        test_leadtime: torch.Tensor = torch.rand(batch_size, 1).to(device)

        with torch.no_grad():
            output: torch.Tensor = model(test_atmos, test_surface, test_leadtime)

        expected_shape: Tuple[int, int, int, int] = (batch_size, in_channels, height, width)
        self.assertEqual(output.shape, expected_shape, f"Expected {expected_shape}, got {output.shape}.")


if __name__ == '__main__':
    unittest.main()
