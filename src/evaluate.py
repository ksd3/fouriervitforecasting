"""
This module provides a function for evaluating a PyTorch model on a given dataset.

It computes the average loss over the dataset using the specified loss function,
while efficiently managing GPU memory.

Functions:
- evaluate: Runs evaluation on a model using a DataLoader and computes the average loss.

Dependencies:
- torch
- tqdm
- gc
"""

import torch
from tqdm import tqdm
import gc
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Union

def evaluate(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    device: Union[str, torch.device]
) -> float:
    """
    Evaluate the model on a given dataset and compute the average loss.
    
    Args:
        model: The PyTorch model to evaluate.
        dataloader: The DataLoader providing evaluation batches.
        criterion: The loss function used for evaluation.
        device: The device on which to perform evaluation (e.g., "cuda" or "cpu").

    Returns:
        float: The average loss over the dataset.
    """
    model.eval()
    total_loss: float = 0.0
    
    with torch.no_grad():
        for (current_atmos, current_surface, lead_time), (target_atmos, target_surface) in tqdm(dataloader, desc="Evaluating"):
            # Move data to the appropriate device
            current_atmos = current_atmos.to(device)
            current_surface = current_surface.to(device)
            lead_time = lead_time.to(device)
            targets = torch.cat([target_atmos.to(device), target_surface.to(device)], dim=1)

            # Forward pass and loss computation
            outputs = model(current_atmos, current_surface, lead_time)
            total_loss += criterion(outputs, targets).item()

            # Clean up to free memory
            del current_atmos, current_surface, lead_time
            del target_atmos, target_surface, outputs, targets
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / len(dataloader)
