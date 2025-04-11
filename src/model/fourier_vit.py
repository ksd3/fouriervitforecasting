"""
This module defines the FourierViT model, which combines Fourier Neural Operator (FNO)
blocks with a Vision Transformer (ViT) architecture for weather forecasting.

The model applies spectral convolutions via FNO blocks to atmospheric and surface
weather data, followed by a transformer-based processing pipeline with patch embeddings.

Classes:
- FourierViT: Combines FNO and ViT components for spatiotemporal weather modeling.

Dependencies:
- torch
- torch.nn
- numpy
- model.fno (FNOBlock)
- model.vit (ViTPatchEmbedding, TransformerBlock, Lambda)
- memory_utils (MemoryProfiler, print_memory_usage)
"""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint

from model.fno import FNOBlock
from model.vit import ViTPatchEmbedding, TransformerBlock, Lambda
from memory_utils import MemoryProfiler, print_memory_usage

class CheckpointedFNOSequential(nn.Sequential):
    """Wrapper for FNO blocks to use checkpointing"""
    def forward(self, x):
        for module in self:
            x = checkpoint.checkpoint(module, x)
        return x

class FourierViT(nn.Module):
    """
    Combines Fourier Neural Operators (FNO) with Vision Transformers (ViT) to process atmospheric and surface data.
    
    Args:
        img_size (Tuple[int, int]): Image dimensions (height, width).
        patch_size (int): Size of the patches used for ViT embedding.
        atmos_vars (int): Number of atmospheric variables.
        atmos_levels (int): Number of atmospheric vertical levels.
        surface_vars (int): Number of surface weather variables.
        embed_dim (int): Embedding dimensionality for ViT.
        fno_modes (Tuple[int, int]): Number of Fourier modes for spectral convolutions.
        fno_width (int): Number of output channels for FNO processing.
        fno_depth (int): Number of FNO blocks in the architecture.
        vit_depth (int): Number of Transformer blocks in ViT.
        vit_heads (int): Number of attention heads in Transformer blocks.
        use_checkpointing (bool): Whether to use activation checkpointing to save memory.
    """
    def __init__(
        self,
        img_size: tuple = (256, 512),
        patch_size: int = 16,
        atmos_vars: int = 7,
        atmos_levels: int = 13,
        surface_vars: int = 7,
        embed_dim: int = 512,
        fno_modes: tuple = (32, 32),
        fno_width: int = 98,
        fno_depth: int = 2,
        vit_depth: int = 4,
        vit_heads: int = 8,
        use_checkpointing: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = (atmos_vars * atmos_levels) + surface_vars
        self.surface_channels = surface_vars
        self.atmos_channels = atmos_vars * atmos_levels
        self.use_checkpointing = use_checkpointing

        # FNO blocks for surface and atmospheric data
        fno_blocks_surface = [
            FNOBlock(
                in_channels=self.surface_channels,
                out_channels=self.surface_channels,
                modes_x=fno_modes[0],
                modes_y=fno_modes[1],
                activation='mish'
            ) for _ in range(fno_depth)
        ]

        fno_blocks_atmos = [
            FNOBlock(
                in_channels=self.atmos_channels,
                out_channels=self.atmos_channels,
                modes_x=16,
                modes_y=16,
                activation='gelu'
            ) for _ in range(fno_depth)
        ]
        
        # Use checkpointing for FNO blocks if enabled
        if use_checkpointing:
            self.surface_fno = CheckpointedFNOSequential(*fno_blocks_surface)
            self.atmos_fno = CheckpointedFNOSequential(*fno_blocks_atmos)
        else:
            self.surface_fno = nn.Sequential(*fno_blocks_surface)
            self.atmos_fno = nn.Sequential(*fno_blocks_atmos)

        # ViT patch embedding
        self.patch_embed = ViTPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=self.in_channels,
            embed_dim=embed_dim
        )

        # Sine activation for lead time encoding
        class SineActivation(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.omega = nn.Parameter(torch.tensor(1.0))
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(self.omega * x)

        self.lead_time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            SineActivation(),
            nn.Linear(embed_dim, embed_dim)
        )
        nn.init.uniform_(self.lead_time_mlp[0].weight, -np.sqrt(6), np.sqrt(6))
        nn.init.zeros_(self.lead_time_mlp[0].bias)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, vit_heads) for _ in range(vit_depth)
        ])

        # Decoder for reconstructing weather predictions
        grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, fno_width),
            nn.GELU(),
            Lambda(lambda x: x.view(x.shape[0], x.shape[1], fno_width)),
            Lambda(lambda x: x.permute(0, 2, 1)),
            Lambda(lambda x: x.reshape(x.shape[0], fno_width, grid_h, grid_w)),
            nn.ConvTranspose2d(
                fno_width, self.in_channels, patch_size, stride=patch_size
            )
        )

    def _run_transformer_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run transformer blocks with optional checkpointing.
        
        Args:
            x: Input tensor to transformer blocks.
            
        Returns:
            torch.Tensor: Output after transformer processing.
        """
        for blk in self.blocks:
            if self.use_checkpointing:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def forward(self, atmos: torch.Tensor, surface: torch.Tensor, lead_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FourierViT model.
        
        Args:
            atmos: Atmospheric data tensor (B, atmos_channels, H, W).
            surface: Surface data tensor (B, surface_channels, H, W).
            lead_time: Lead time tensor (B, 1).
        
        Returns:
            torch.Tensor: Predicted weather conditions of shape (B, in_channels, H, W).
        """
        with MemoryProfiler() as mp:
            x = torch.cat([atmos, surface], dim=1)
            atmos_part, surface_part = x[:, :self.atmos_channels], x[:, self.atmos_channels:]
            print_memory_usage("After input concat")

            atmos_out = self.atmos_fno(atmos_part) + atmos_part
            surface_out = self.surface_fno(surface_part) + surface_part
            print_memory_usage("After FNO blocks")

            x_fno = torch.cat([atmos_out, surface_out], dim=1)
            x_patches = self.patch_embed(x_fno)
            print_memory_usage("After patch embedding")

            lt_emb = self.lead_time_mlp(lead_time).unsqueeze(1)
            x_patches = torch.cat([lt_emb, x_patches], dim=1)

            # Use the checkpointed transformer blocks
            x_patches = self._run_transformer_blocks(x_patches)
            print_memory_usage("After transformer")

            x_out = self.decoder(x_patches[:, 1:, :])
            print_memory_usage("After decoder")

        total_mb = (mp.end - mp.begin) / 1e9
        print(f"Total FWD memory: {total_mb:.2f}GB")
        return x_out