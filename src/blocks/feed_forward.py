import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            bias: bool = True,
            norm: Optional[Tuple] = None
    ):
        '''
        dim: int = channels in input and output tensor
        bias: bool = use bias in linear modules
        norm: Tuple = optional normalized shape for LayerNorm operation; if None skip normalization (not recommended)
        '''

        super(FeedForward, self).__init__()

        self.lin1 = nn.Linear(dim, dim, bias=bias)
        self.lin2 = nn.Linear(dim, dim, bias=bias)

        if norm:
            self.norm = nn.LayerNorm(normalized_shape=norm, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin1(x)
        h = F.gelu(h)
        h = self.lin2(h)
        h = h + x
        if hasattr(self, 'norm'):
            h = self.norm(h)
        return h