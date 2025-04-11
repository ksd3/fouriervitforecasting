"""
This module contains the implementation of a single training epoch for a PyTorch model.
It includes data loading, forward propagation, loss computation, backpropagation, and optimizer updates.
The function is designed to handle GPU memory efficiently with garbage collection and emptying CUDA caches.

Functions:
- train_epoch: Runs one training epoch on the given model and dataset.

Dependencies:
- torch
- gc
- tqdm
- torch.nn.Module
- torch.optim.Optimizer
- torch.utils.data.DataLoader
"""

import torch
import gc
from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Union

def train_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: Union[str, torch.device],
    epoch: int
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: The DataLoader providing training batches.
        optimizer: The optimizer used for training.
        criterion: The loss function used for optimization.
        device: The device on which to perform training (e.g., "cuda" or "cpu").
        epoch: The current epoch number, for logging purposes.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss: float = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for (current_atmos, current_surface, lead_time), (target_atmos, target_surface) in pbar:
        # Move data to the appropriate device
        current_atmos = current_atmos.to(device)
        current_surface = current_surface.to(device)
        lead_time = lead_time.to(device)
        target_atmos = target_atmos.to(device)
        target_surface = target_surface.to(device)

        # Forward pass
        outputs = model(current_atmos, current_surface, lead_time)
        targets = torch.cat([target_atmos, target_surface], dim=1)

        # Compute loss
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Clean up to free memory
        del current_atmos, current_surface, lead_time
        del target_atmos, target_surface, outputs, targets
        torch.cuda.empty_cache()
        gc.collect()

    return total_loss / len(dataloader)
