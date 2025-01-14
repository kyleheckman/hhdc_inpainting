import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MaskedConv(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int,
            stride: int = 1,
            bias: bool = False,
            norm = None
    ):
        super(MaskedConv, self).__init__()

        self.stride = stride
        ''' Calculate padding to keep same dimensions if stride == 1 or
        return dim//stride if stride > 1 '''
        conv_padding = (kernel_size-1)//2
        self.pad = nn.ZeroPad2d(conv_padding)
    
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding='valid', bias=bias)
        self.conv_mask = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding='valid', bias=bias)
        # Set kernel weight for mask convolution to all 1s
        #self.conv_mask.weight = torch.nn.Parameter(torch.ones_like(self.conv_mask.weight))

        self.kernel = torch.ones(out_ch, in_ch, kernel_size, kernel_size)

        if norm is not None:
            self.norm = nn.LayerNorm(normalized_shape=norm, bias=bias)
    
    def forward(self, x, mask):
        x = self.pad(x)
        mask = self.pad(mask)
        if x.shape[-2:] != mask.shape[-2:]:
            raise Exception(f'Input and Mask must be of the same Height and Width. Got input:{x.shape} and mask:{mask.shape}')
        
        x = self.conv(x*mask)
        #mask = self.conv_mask(mask)
        mask = F.conv2d(mask, self.kernel, bias=None)
        mask = torch.clamp(mask, min=0, max=1)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = F.silu(x)

        return x, mask

class MaskedConv_2Block(nn.Module):
    def __init__(
            self,
            channels: List,
            kernel_size: int,
            stride: int = 1,
            bias: bool = True,
            norm = None
    ):
        super(MaskedConv_2Block, self).__init__()

        if len(channels) != 3:
            raise Exception(f'Parameter [channels] requires a list of size 3, got {channels}')

        self.stride = stride
        ''' Calculate padding to keep same dimensions if stride == 1 or
        return dim//stride if stride > 1 '''
        conv_padding = (kernel_size-1)//2
        self.pad = nn.ReplicationPad2d(conv_padding)

        self.window_shape = kernel_size**2

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=1, padding='valid', bias=bias)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, stride=stride, padding='valid', bias=bias)

        self.kern = torch.ones(channels[2], channels[0], kernel_size, kernel_size).to('cuda')

        if norm:
            self.norm = nn.LayerNorm(normalized_shape=norm, bias=bias)
    
    def forward(self, x, mask):
        if x.shape[-2:] != mask.shape[-2:]:
            raise Exception(f'Input and Mask must be of the same Height and Width. Got input:{x.shape} and mask:{mask.shape}')
        
        x = self.pad(x)
        mask = self.pad(mask)

        x = self.conv2(self.pad(F.silu(self.conv1(x*mask))))
        mask = F.conv2d(mask, self.kern, bias=None, stride=self.stride)

        mask = torch.clamp(mask, min=0, max=1)

        #x = self.pad(x)
        #mask = self.pad(mask)

        #x = self.conv2(x)
        #mask = F.conv2d(mask, self.kern2, bias=None, stride=self.stride)
        
        #mask = torch.clamp(mask, min=0, max=1)

        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = F.silu(x)

        return x, mask