"""
This module implements core Transformer components including self-attention,
feed-forward layers, and patch embedding for Vision Transformers (ViT).

Classes:
- MultiHeadSelfAttention: Implements multi-head self-attention for Transformers.
- TransformerBlock: Defines a standard Transformer block with attention and MLP layers.
- ViTPatchEmbedding: Converts input images into patch embeddings for ViT models.
- Lambda: Wraps a function into an nn.Module for functional transformations.

Dependencies:
- torch
- torch.nn
- numpy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention for Transformers.
    
    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, D), where B=batch size, N=sequence length, D=embedding dim.
        
        Returns:
            torch.Tensor: Output tensor after self-attention.
        """
        B, N, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """
    Defines a standard Transformer block with self-attention and an MLP layer.
    
    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Expansion ratio for the MLP layer.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.
        
        Args:
            x: Input tensor of shape (B, N, D).
        
        Returns:
            torch.Tensor: Output tensor after attention and feed-forward network.
        """
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ViTPatchEmbedding(nn.Module):
    """
    Converts an image into patch embeddings for Vision Transformers (ViT).
    
    Args:
        img_size (Tuple[int, int]): Input image dimensions.
        patch_size (int): Size of patches extracted from the image.
        in_channels (int): Number of input channels.
        embed_dim (int): Dimensionality of the patch embeddings.
    """
    def __init__(
        self, img_size: Tuple[int, int] = (256, 512), patch_size: int = 16, in_channels: int = 98, embed_dim: int = 512
    ) -> None:
        super().__init__()
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch embedding module.
        
        Args:
            x: Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        x = self.proj(x).reshape(B, self.proj.out_channels, -1).transpose(1, 2)
        return x + self.pos_embed

class Lambda(nn.Module):
    """
    A wrapper module for applying a custom function within an nn.Module.
    
    Args:
        func (Callable): A function to apply to the input tensor.
    """
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the wrapped function to the input tensor.
        
        Args:
            x: Input tensor.
        
        Returns:
            torch.Tensor: Processed tensor after applying the function.
        """
        return self.func(x)
