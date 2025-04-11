"""
This module provides utilities for monitoring GPU memory usage in PyTorch.

It includes functions to print real-time memory consumption and a context manager
for profiling memory usage during code execution.

Functions:
- print_memory_usage: Prints the current GPU memory allocated.

Classes:
- MemoryProfiler: A context manager that tracks memory allocation between operations.

Dependencies:
- torch
"""

import torch

def print_memory_usage(msg: str = "") -> None:
    """
    Prints the amount of GPU memory currently allocated.
    
    Args:
        msg: An optional message to display alongside the memory usage.
    """
    if torch.cuda.is_available():
        allocated: float = torch.cuda.memory_allocated() / 1e9  # Convert bytes to gigabytes
        print(f"{msg}: {allocated:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

class MemoryProfiler:
    """
    A context manager for measuring GPU memory usage during a block of code execution.
    
    Usage:
        with MemoryProfiler():
            # Code block to monitor GPU memory usage
    """
    def __enter__(self) -> "MemoryProfiler":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.begin: int = torch.cuda.memory_allocated()
        else:
            self.begin = 0
        return self

    def __exit__(self, *args: object) -> None:
        if torch.cuda.is_available():
            self.end: int = torch.cuda.memory_allocated()
            delta: float = (self.end - self.begin) / 1e9  # Convert bytes to gigabytes
            print(f"Delta: {delta:.2f}GB")
