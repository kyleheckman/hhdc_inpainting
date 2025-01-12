import torch
import torch.nn as nn
import torch.nn.functional as F

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

        ''' Calculate padding to keep same dimensions if stride == 1 or
        return dim//stride if stride > 1 '''
        conv_padding = (kernel_size-1)//2
        self.pad = nn.ZeroPad2d(conv_padding)
    
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding='valid', bias=bias)
        self.conv_mask = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding='valid', bias=bias)
        # Set kernel weight for mask convolution to all 1s
        self.conv_mask.weight = torch.nn.Parameter(torch.ones_like(self.conv_mask.weight))

        if norm is not None:
            self.norm = nn.LayerNorm(normalized_shape=norm, bias=bias)
    
    def forward(self, x, mask):
        x = self.pad(x)
        mask = self.pad(mask)
        if x.shape[-2:] != mask.shape[-2:]:
            raise Exception(f'Input and Mask must be of the same Height and Width. Got input:{x.shape} and mask:{mask.shape}')
        
        x = self.conv(x*mask)
        mask = self.conv_mask(mask)
        mask = torch.clamp(mask, min=0, max=1)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = F.silu(x)

        return x, mask
