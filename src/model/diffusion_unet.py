import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from ..blocks.resnet import ResTimeEmbBlock2d
from ..blocks.transformer import Transformer2d
from ..utils.positional_encoding import SinusoidalTimeEmbeddings

class DiffusionUnet(nn.Module):
    def __init__(
            self,
            time_steps: int = 1000,
            bias: bool = True
    ):
        '''
        time_steps: int = number of time steps in noise schedule
        bias: bool = use bias in modules
        '''

        super(DiffusionUnet, self).__init__()

        self.in_ref = nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode='zeros', bias=bias)

        self.enc11 = ResTimeEmbBlock2d(in_channels=32, out_channels=64, norm=(128,64), bias=bias)
        #self.enc12 = ResTimeEmbBlock2d(in_channels=64, out_channels=64, norm=(128,64), bias=bias)
        self.pre_norm1 = nn.LayerNorm(normalized_shape=(128,64), bias=bias)
        self.sattn1 = Transformer2d(block_size=(32,32), embed_dim=768, seq_dim=512, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)

        self.down_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='zeros', bias=bias)
        self.enc21 = ResTimeEmbBlock2d(in_channels=64, out_channels=128, norm=(64,32), bias=bias)
        #self.enc22 = ResTimeEmbBlock2d(in_channels=128, out_channels=128, norm=(64,32), bias=bias)
        self.pre_norm2 = nn.LayerNorm(normalized_shape=(64,32), bias=bias)
        self.sattn2 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)
        
        self.down_conv2= nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, padding_mode='zeros', bias=bias)
        self.enc31 = ResTimeEmbBlock2d(in_channels=128, out_channels=256, norm=(32,16), bias=bias)
        #self.enc32 = ResTimeEmbBlock2d(in_channels=256, out_channels=256, norm=(32,16), bias=bias)
        self.pre_norm3 = nn.LayerNorm(normalized_shape=(32,16), bias=bias)
        self.sattn3 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=512, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)
        
        self.down_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, padding_mode='zeros', bias=bias)
        self.mid1 = ResTimeEmbBlock2d(in_channels=256, out_channels=512, norm=(16,8), bias=bias)
        self.pre_norm4 = nn.LayerNorm(normalized_shape=(16,8), bias=bias)
        #self.mid2 = ResTimeEmbBlock2d(in_channels=512, out_channels=512, norm=(16,8), bias=bias)
        self.sattn4 = Transformer2d(block_size=(8,8), embed_dim=768, seq_dim=1024, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)
        self.mid3 = ResTimeEmbBlock2d(in_channels=512, out_channels=256, norm=(16,8), bias=bias)

        self.up_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=bias)
        self.dec31 = ResTimeEmbBlock2d(in_channels=512, mid_channels=256, out_channels=128, norm=(32,16), bias=bias)
        #self.dec32 = ResTimeEmbBlock2d(in_channels=256, out_channels=128, norm=(32,16), bias=bias)
        self.pre_norm5 = nn.LayerNorm(normalized_shape=(32,16),bias=bias)
        self.sattn5 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=512, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)
        
        self.up_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=bias)
        self.dec21 = ResTimeEmbBlock2d(in_channels=256, mid_channels=128, out_channels=64, norm=(64,32), bias=bias)
        #self.dec22 = ResTimeEmbBlock2d(in_channels=128, out_channels=64, norm=(64,32), bias=bias)
        self.pre_norm6 = nn.LayerNorm(normalized_shape=(64,32), bias=bias)
        self.sattn6 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)
        
        self.up_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=bias)
        self.dec11 = ResTimeEmbBlock2d(in_channels=128, mid_channels=64, out_channels=32, norm=(128,64), bias=bias)
        #self.dec12 = ResTimeEmbBlock2d(in_channels=64, out_channels=32, norm=(128,64), bias=bias)
        self.pre_norm7 = nn.LayerNorm(normalized_shape=(128,64), bias=bias)
        self.sattn7 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, attn_dropout=0.1, norm=(768), bias=bias)

        self.out_norm = nn.LayerNorm(normalized_shape=(128,64))
        self.out_ref = nn.Conv2d(32, 1, kernel_size=1)

        self.time_emb = SinusoidalTimeEmbeddings(time_steps=time_steps, max_dim=512)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, t2i: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if t2i:
            ti1, ti2, ti3, ti4 = t2i
        embed = self.time_emb(x, t)

        h = self.in_ref(x)

        if ti1:
            h = h + ti1

        e1 = self.enc11(h, embed)
        e1 = e1 + self.sattn1(self.pre_norm1(e1))

        e2 = self.down_conv1(e1)
        if ti2:
            e2 = e2 + ti2
        e2 = self.enc21(e2, embed)
        e2 = e2 + self.sattn2(self.pre_norm2(e2))

        e3= self.down_conv2(e2)
        if ti3:
            e3 = e3 + ti3
        e3 = self.enc31(e3, embed)
        e3 = e3 + self.sattn3(self.pre_norm3(e3))

        z = self.down_conv3(e3)
        if ti4:
            z = z + ti4
        z = self.mid1(z, embed)
        z = z + self.sattn4(self.pre_norm4(z))
        z = self.mid3(z, embed)

        z = self.up_conv3(z)
        z = torch.concat((z, e3), axis=1)
        z = self.dec31(z, embed)
        z = z + self.sattn5(self.pre_norm5(z))

        z = self.up_conv2(z)
        z = torch.concat((z, e2), axis=1)
        z = self.dec21(z, embed)
        z = z + self.sattn6(self.pre_norm6(z))

        z = self.up_conv1(z)
        z = torch.concat((z, e1), axis=1)
        z = self.dec11(z, embed)
        z = z + self.sattn7(self.pre_norm7(z))

        return self.out_ref(z)



