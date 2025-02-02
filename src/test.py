import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.resnet import ResTimeEmbBlock2d
from .utils.positional_encoding import SinusoidalTimeEmbeddings
from .blocks.transformer import Transformer2d

if __name__ == '__main__':
    x = torch.rand(1,1,128,64)
    t = torch.Tensor([1]).to(torch.int)

    time_emb = SinusoidalTimeEmbeddings(time_steps=4, max_dim=512)

    in_ref = nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode='zeros')

    enc1 = ResTimeEmbBlock2d(in_channels=32, out_channels=64, norm=(128,64))
    sattn1 = Transformer2d(block_size=(32,32), embed_dim=768, seq_dim=512, attn_num_heads=8, norm=(768))

    down_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
    enc2 = ResTimeEmbBlock2d(in_channels=64, out_channels=128, norm=(64,32))
    sattn2 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, norm=(768))

    down_conv2 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
    enc3 = ResTimeEmbBlock2d(in_channels=128, out_channels=256, norm=(32,16))
    sattn3 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=512, attn_num_heads=8, norm=(768))

    down_conv3 = nn.Conv2d(256,256, kernel_size=3,stride=2, padding=1, padding_mode='zeros')
    mid1 = ResTimeEmbBlock2d(in_channels=256, out_channels=512, norm=(16,8))
    sattn4 = Transformer2d(block_size=(8,8), embed_dim=768, seq_dim=1024, attn_num_heads=8, norm=(768))
    mid3 = ResTimeEmbBlock2d(in_channels=512, out_channels=256, norm=(16,8))

    up_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    dec3 = ResTimeEmbBlock2d(in_channels=512, mid_channels=256, out_channels=128, norm=(32,16))
    sattn5 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=512, attn_num_heads=8, norm=(768))

    up_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
    dec2 = ResTimeEmbBlock2d(in_channels=256, mid_channels=128, out_channels=64, norm=(64,32))
    sattn6 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, norm=(768))

    up_conv1 = nn.ConvTranspose2d(64,64, kernel_size=2, stride=2)
    dec1 = ResTimeEmbBlock2d(in_channels=128, mid_channels=64, out_channels=32, norm=(128,64))
    sattn7 = Transformer2d(block_size=(16,16), embed_dim=768, seq_dim=1024, attn_num_heads=8, norm=(768))

    out_ref = nn.Conv2d(32, 1, kernel_size=1)
    
    print(f'X {x.shape}')
    h = in_ref(x)
    embed = time_emb(x, t)
    e1 = enc1(h, embed)
    attn1 = sattn1(e1)
    e1 = e1 + attn1
    print(f'E1 FINAL {e1.shape}')

    e2 = down_conv1(e1)
    print(f'DOWN {e2.shape}')
    e2 = enc2(e2, embed)
    attn2 = sattn2(e2)
    e2 = e2 + attn2
    print(f'E2 FINAL {e2.shape}')

    e3 = down_conv2(e2)
    print(f'DOWN {e3.shape}')
    e3 = enc3(e3, embed)
    attn3 = sattn3(e3)
    e3 = e3 + attn3
    print(f'E3 FINAL {e3.shape}')

    z = down_conv3(e3)
    print(f'DOWN {z.shape}')
    z = mid1(z, embed)
    attn4 = sattn4(z)
    z = z + attn4
    print(f'MID1 SA {z.shape}')
    z = mid3(z, embed)
    print(f'MID FINAL {z.shape}')

    z = up_conv3(z)
    print(f'UP {z.shape}')
    z = torch.concat((z, e3), axis=1)
    print(f'CONCAT {z.shape}')
    z = dec3(z, embed)
    attn5 = sattn5(z)
    z = z + attn5
    print(f'Z FINAL {z.shape}')

    z = up_conv2(z)
    z = torch.concat((z, e2), axis=1)
    z = dec2(z, embed)
    attn6 = sattn6(z)
    z = z + attn6

    z = up_conv1(z)
    z = torch.concat((z, e1), axis=1)
    z = dec1(z, embed)
    attn7 = sattn7(z)
    z = z + attn7

    z = out_ref(z)
    print(f'FINAL {z.shape}')