"""
This module initializes key components of the model by importing essential layers and architectures.
It includes Fourier Neural Operator (FNO) layers, Transformer-based components, and the
FourierViT model, which combines FNO and Vision Transformer architectures.

Imports:
- SpectralConv2d: Spectral convolutional layer from FNO.
- FNOBlock: Fourier Neural Operator block.
- MultiHeadSelfAttention: Transformer self-attention mechanism.
- TransformerBlock: Standard Transformer block.
- ViTPatchEmbedding: Patch embedding for Vision Transformers.
- FourierViT: Full model combining FNO and ViT components.

Dependencies:
- fno
- vit
- fourier_vit
"""

from .fno import SpectralConv2d, FNOBlock
from .vit import MultiHeadSelfAttention, TransformerBlock, ViTPatchEmbedding
from .fourier_vit import FourierViT
