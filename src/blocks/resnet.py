import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ResTimeEmbBlock2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            dilation: int = 1,
            padding: int = 1,
            padding_mode: Optional[str] = None,
            bias: bool = True,
            dropout: float = 0.0,
            norm: Optional[Tuple] = None
    ):
        '''
        in_channels: int = channels of input tensor
        out_channels: int = channels of output tensor
        mid_channels: int = optional number of channels of intermediate tensor
        kernel_size: int = kernel size for convolution modules
        dilation: int = dilation for convolutional modules
        padding: int = padding on convolutional modules
        padding_mode: str = padding mode use in convolutional layers, defaults to zeros
        bias: bool = use bias in layers
        dropout: float = dropout probability for scaled_dot_product_attention
        norm: Tuple = normalized shape in LayerNorm, if None then normalization is skipped (not recommended)
        '''
        
        super(ResTimeEmbBlock2d, self).__init__()

        self.in_channels = in_channels
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

        padding_mode = 'zeros' if padding_mode is None else padding_mode
        self.padding_mode = padding_mode

        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, padding_mode=padding_mode, bias=bias)

        self.dropout = nn.Dropout(dropout)

        if norm:
            self.norm1 = nn.LayerNorm(normalized_shape=norm, bias=bias)
            self.norm2 = nn.LayerNorm(normalized_shape=norm, bias=bias)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x

        if hasattr(self, 'norm1'):
            h = self.norm1(h)
        h = self.conv1(F.gelu(h))

        h = h + t_emb[:,:h.shape[1],:,:]

        if hasattr(self, 'norm2'):
            h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.conv2(h)

        return h + self.shortcut(x)

class ResBlock2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            dilation: int = 1,
            padding: int = 1,
            padding_mode: Optional[str] = None,
            bias: bool = True,
            dropout: float = 0.0,
            norm: Optional[Tuple] = None
    ):
        '''
        in_channels: int = channels of input tensor
        out_channels: int = channels of output tensor
        mid_channels: int = optional number of channels of intermediate tensor
        kernel_size: int = kernel size for convolution modules
        dilation: int = dilation for convolutional modules
        padding: int = padding on convolutional modules
        padding_mode: str = padding mode use in convolutional layers, defaults to zeros
        bias: bool = use bias in layers
        dropout: float = dropout probability for scaled_dot_product_attention
        norm: Tuple = normalized shape in LayerNorm, if None then normalization is skipped (not recommended)
        '''
        
        super(ResBlock2d, self).__init__()

        self.in_channels = in_channels
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

        padding_mode = 'zeros' if padding_mode is None else padding_mode
        self.padding_mode = padding_mode

        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, padding_mode=padding_mode, bias=bias)

        if norm:
            self.norm1 = nn.LayerNorm(normalized_shape=norm)
            self.norm2 = nn.LayerNorm(normalized_shape=norm)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        if hasattr(self, 'norm'):
            h = self.norm1(h)
        h = self.conv1(F.gelu(h))

        if hasattr(self, 'norm2'):
            h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.conv2(h)

        return h + self.shortcut(x)
