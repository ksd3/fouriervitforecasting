import torch
import torch.nn as nn
import numpy as np

def complx_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise complex multiplication in 2D.
    
    Args:
        a: A complex-valued tensor with last dimension size 2 (real and imaginary parts).
        b: A complex-valued tensor with last dimension size 2 (real and imaginary parts).
    
    Returns:
        A tensor containing the result of the complex multiplication.
    """
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack([real, imag], dim=-1)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, inC, H, W = x.shape
        # Explicitly cast x to FP32 for the FFT operations.
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfft2(x_fp32, norm='ortho')
        x_ft = torch.view_as_real(x_ft)
        # Create out_ft as FP32 regardless of x's original dtype.
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[2], x_ft.shape[3], 2,
                            device=x.device, dtype=torch.float32)
        
        x_limit = min(self.modes_x, x_ft.shape[2])
        y_limit = min(self.modes_y, x_ft.shape[3])
        
        x_ft_slice = x_ft[:, :, :x_limit, :y_limit, :]
        w = self.weights[:, :, :x_limit, :y_limit, :]
        
        x_ft_slice = x_ft_slice.unsqueeze(2)
        w = w.unsqueeze(0)
        out_slice = complx_mul2d(x_ft_slice, w)
        out_slice = out_slice.sum(dim=1)
        out_ft[:, :, :x_limit, :y_limit, :] = out_slice
        
        out_ft_complex = torch.view_as_complex(out_ft)
        x_out = torch.fft.irfft2(out_ft_complex, s=(H, W), norm='ortho')
        return x_out



class FNOBlock(nn.Module):
    """
    Defines a Fourier Neural Operator (FNO) block that applies spectral and linear convolutions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes_x: Number of Fourier modes retained along the x-axis.
        modes_y: Number of Fourier modes retained along the y-axis.
        activation: Activation function to use ('swish', 'gelu', 'mish').
    """
    def __init__(
        self, in_channels: int, out_channels: int, modes_x: int = 16, modes_y: int = 16, activation: str = 'swish'
    ) -> None:
        super().__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, modes_x, modes_y)
        self.w = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Linear 1x1 convolution
        
        # Select activation function
        if activation == 'swish':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'mish':
            self.act = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FNO block.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W).
        
        Returns:
            Output tensor after spectral and linear transformations followed by activation.
        """
        out_spectral = self.conv(x)
        out_linear = self.w(x)
        return self.act(out_spectral + out_linear)
