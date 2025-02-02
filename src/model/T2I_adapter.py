import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..blocks.resnet import ResBlock2d

class T2IAdapter(nn.Module):
    def __init__(
            self,
            num_layers: int,
            channels: List,
            norms: List[Tuple],
            bias: bool = True
    ):
        
        module_list = []

        for layer in range(num_layers):
            module = nn.Conv2d(channels[layer], channels[layer], kernel_size=3, padding=1, padding_mode='zeros', bias=bias)
            module_list.append(module)

            module = ResBlock2d(in_channels=channels[layer], out_channels=channels[layer], norm=norms[layer])
            module_list.append(module)

            module = ResBlock2d(in_channels=channels[layer], out_channels=channels[layer], norm=norms[layer])
            module_list.append(module)
        