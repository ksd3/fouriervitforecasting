"""
This module provides an example of multi-GPU training using PyTorch's
Fully Sharded Data Parallel (FSDP) for the FourierViT model.

You can launch this script via:
    torchrun --nproc_per_node=<NUM_GPUS> train_distributed.py

Replace <NUM_GPUS> with the number of GPUs you want to utilize on a single node.
For a single GPU test, you can try:
    torchrun --nproc_per_node=1 train_distributed.py
Though that won't actually achieve parallel speedup, it should still run.

IMPORTANT:
- Make sure you have installed PyTorch with FSDP support and have NCCL, etc.
- This file assumes you have the same directory structure as before, with:
    config.py, dataset.py, memory_utils.py, model/..., train.py, evaluate.py
  in the same folder. We specifically import FourierViT, WeatherSubset, etc.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch._dynamo

from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.fsdp import MixedPrecision
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Tuple, Optional
import argparse

# Local imports from our prior code structure
from config import config
from dataset import WeatherSubset
from model.fourier_vit import FourierViT

torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = True  

def is_bf16_supported() -> bool:
    """Check if BF16 is supported on the current device."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

def train_epoch_distributed(
    model: FSDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
    dtype: Optional[torch.dtype] = None
) -> float:
    """
    Perform one epoch of training in a distributed fashion.
    
    Args:
        model: The FSDP-wrapped model.
        dataloader: The distributed DataLoader.
        optimizer: The optimizer for training.
        criterion: Loss function to calculate MSE, cross-entropy, etc.
        device: The current device (GPU).
        epoch: The current epoch number, used for logging.
        rank: The process rank (useful for printing only in rank=0).
        dtype: The data type to use for mixed precision training.
    
    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False) if rank == 0 else dataloader

    for _, ((current_atmos, current_surface, lead_time), (target_atmos, target_surface)) in enumerate(pbar):
        current_atmos, current_surface, lead_time = (
            current_atmos.to(device), current_surface.to(device), lead_time.to(device)
        )
        target_atmos, target_surface = target_atmos.to(device), target_surface.to(device)
        targets = torch.cat([target_atmos, target_surface], dim=1)

        optimizer.zero_grad()
        
        # Use the new autocast API
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=dtype is not None):
            outputs = model(current_atmos, current_surface, lead_time)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        del current_atmos, current_surface, lead_time, target_atmos, target_surface, outputs, targets
        torch.cuda.empty_cache()
        gc.collect()

    total_loss_tensor = torch.tensor([total_loss], device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / dist.get_world_size()

    return avg_loss / len(dataloader)

def evaluate_distributed(
    model: FSDP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    dtype: Optional[torch.dtype] = None
) -> float:
    """
    Evaluate the model in a distributed fashion.
    
    Args:
        model: The FSDP-wrapped model in eval mode.
        dataloader: The distributed DataLoader for evaluation.
        criterion: Loss function (e.g., MSE).
        device: The current device (GPU).
        rank: The process rank.
        dtype: The data type to use for mixed precision evaluation.
    
    Returns:
        float: The average evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Evaluating", leave=False) if rank == 0 else dataloader

    with torch.no_grad():
        for (current_atmos, current_surface, lead_time), (target_atmos, target_surface) in pbar:
            current_atmos, current_surface, lead_time = (
                current_atmos.to(device), current_surface.to(device), lead_time.to(device)
            )
            targets = torch.cat([target_atmos.to(device), target_surface.to(device)], dim=1)

            # Use the new autocast API
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=dtype is not None):
                outputs = model(current_atmos, current_surface, lead_time)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item()

            del current_atmos, current_surface, lead_time, target_atmos, target_surface, outputs, targets
            torch.cuda.empty_cache()
            gc.collect()

    total_loss_tensor = torch.tensor([total_loss], device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / dist.get_world_size()
    return avg_loss / len(dataloader)

def main() -> None:
    """
    Main function to initialize the distributed environment, set up the FSDP-wrapped
    FourierViT model, and run multi-GPU training + evaluation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed training for FourierViT")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 precision even if BF16 is supported")
    args, unknown_args = parser.parse_known_args()
    # Fix for running directly without torchrun
    if "LOCAL_RANK" not in os.environ:
        print("Running in non-distributed mode (single GPU)")
        local_rank = 0
        world_size = 1
        rank = 0
        # Set environment variables for PyTorch distributed
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        # Initialize process group with a single process
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend="nccl")
    
    device = torch.device("cuda", local_rank)
    
    # Check if BF16 is supported and set up mixed precision
    bf16_supported = is_bf16_supported()
    if bf16_supported:
        dtype = torch.bfloat16
        if rank == 0:
            print("Using BF16 precision")
    else:
        dtype = None
        if rank == 0:
            print("BF16 not supported, using FP32")
    
    # Set up mixed precision policy for FSDP if BF16 is supported
    if bf16_supported:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    else:
        mixed_precision_policy = None

    dataset = WeatherSubset(config['data_path'], config['max_data_mb'])
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=2, pin_memory=True)

    # Create model with activation checkpointing enabled
    model = FourierViT(
        img_size=(256,512), patch_size=16, atmos_vars=len(config['atmos_vars']),
        atmos_levels=config['atmos_levels'], surface_vars=len(config['surface_vars']),
        embed_dim=512, fno_modes=(32,32), fno_width=98, fno_depth=2, vit_depth=4, vit_heads=8,
        use_checkpointing=True  # Enable activation checkpointing
    ).to(device)

    # Fix the FSDP configuration to work with torch.compile
    use_compile = hasattr(torch, 'compile') and torch.__version__ >= "2.0.0"
    
    # Wrap model with FSDP and mixed precision if supported
    model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        use_orig_params=True  # Required for torch.compile compatibility
    )
    
    # Apply torch.compile after FSDP wrapping
    if use_compile:
        # Suppress errors if needed
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
        if rank == 0:
            print("Using torch.compile")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        epoch_loss = train_epoch_distributed(model, dataloader, optimizer, criterion, device, epoch, rank, dtype)
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {epoch_loss:.4f}")

    val_loss = evaluate_distributed(model, dataloader, criterion, device, rank, dtype)
    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}")
    
    if rank == 0:
        # Save model in BF16 format if used
        model_filename = "fourier_vit_distributed_bf16.pth" if bf16_supported else "fourier_vit_distributed.pth"
        torch.save(model.state_dict(), model_filename)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()