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
        super(T2IAdapter, self).__init__()


        self.module_list = nn.ModuleList([])

        self.in_ref = nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode='zeros', bias=bias)

        for layer in range(num_layers):
            module = nn.Conv2d(channels[layer], channels[layer], kernel_size=3, padding=1, padding_mode='zeros', bias=bias)
            self.module_list.append(module)

            module = ResBlock2d(in_channels=channels[layer], out_channels=channels[layer], norm=norms[layer])
            self.module_list.append(module)

            module = ResBlock2d(in_channels=channels[layer], out_channels=channels[layer], norm=norms[layer])
            self.module_list.append(module)

            self.module_list.append(nn.LayerNorm(normalized_shape=norms[layer]))
            
            # If not last layer, downsample using pixel unshuffle
            if layer != num_layers-1:
                self.module_list.append(nn.PixelUnshuffle(2))
                self.module_list.append(nn.Conv2d(channels[layer]*4, channels[layer+1], kernel_size=1, bias=bias))

    def forward(self, x: torch.Tensor, intm_out: List[int]) -> torch.Tensor:
        temp = []
        
        h = F.gelu(self.in_ref(x))
        for i, module in enumerate(self.module_list):
            h = module(h)
            if i in intm_out:
                temp.append(h)
        temp.append(h)
        return temp


if __name__ == '__main__':
    x = torch.rand(1,1,128,64)

    adapt = T2IAdapter(num_layers=4, channels=[32,64,128,256], norms=[(128,64),(64,32),(32,16),(16,8)])
    temp = adapt(x, [3,9,15])

    print(len(temp))
    for i in temp:
        print(i.shape)