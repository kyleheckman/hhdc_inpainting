import torch
import torch.nn as nn
import torch.nn.functional as F
from masked_conv import MaskedConv_2Block

class Autoencoder(nn.Module):
    def __init__(
            self,
            feat_ch: int = 32
    ):
        super(Autoencoder, self).__init__()

        self.e0 = MaskedConv_2Block([1,16,32], kernel_size=7, interpolate=True)                      # (b, 1, 32, 32) -> (b, 32, 32, 32)
        self.e1 = MaskedConv_2Block([32,64,64], kernel_size=5, stride=2)          # (b, 32, 32, 32) -> (b, 64, 16, 16)
        self.e2 = MaskedConv_2Block([64,128,128], kernel_size=3, stride=2)        # (b, 64, 16, 16) -> (b, 128, 8, 8)
        self.e3 = MaskedConv_2Block([128,256,256], kernel_size=3, stride=2)       # (b, 128, 8, 8) -> (b, 256, 4, 4)
        self.e4 = MaskedConv_2Block([256,256,256], kernel_size=3)                 # (b, 256, 4, 4) -> (b, 256, 4, 4)

        self.dense_mean = nn.Linear(4096, 3072)
        self.dense_logvar = nn.Linear(4096, 3072)

        self.dense_dec = nn.Linear(3072, 4096)

        self.d4 = MaskedConv_2Block([256,256,256], kernel_size=3)                 # (b, 256, 4, 4) -> (b, 256, 4, 4)
        self.d3 = MaskedConv_2Block([512,256,128], kernel_size=3)                 # (b, 512, 8, 8) -> (b, 128, 8, 8)
        self.d2 = MaskedConv_2Block([256,128,64], kernel_size=3)                  # (b, 256, 16, 16) -> (b, 64, 16, 16)
        self.d1 = MaskedConv_2Block([128,64,32], kernel_size=3)                  # (b, 128, 32, 32) -> (b, 32, 32, 32)
        self.d0 = MaskedConv_2Block([64,32,8], kernel_size=5)                    # (b, 64, 32, 32) -> (b, 3, 32, 32)

        self.out_ref = nn.Conv2d(8, 1, kernel_size=1, padding='same', padding_mode='replicate', bias=False)   # (b, 3, 32, 32) -> (b, 1, 32, 32)

        self.upscale = nn.Upsample(scale_factor=2)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mean + (std * eps)

    def encode(self, x, mask):
        residuals = []

        # (1, 32, 32) -> (32, 32, 32)
        x, mask = self.e0(x, mask)
        residuals.append((x, mask))

        # (32, 32, 32) -> (64, 16, 16)
        x, mask = self.e1(x, mask)
        residuals.append((x, mask))

        # (64, 16, 16) -> (128, 8, 8)
        x, mask = self.e2(x, mask)
        residuals.append((x, mask))

        # (128, 8, 8) -> (256, 4, 4)
        x, mask = self.e3(x, mask)
        residuals.append((x, mask))

        # (256, 4, 4) -> (256, 4, 4)
        x, mask = self.e4(x, mask)
        residuals.append((x, mask))

        # Flatten image values and pass to dense layers
        x = x.view(x.shape[0], -1)      # (b, 4096)
        #x = F.silu(self.dense_e5(x))    # (b, 3072)
        #x = F.silu(self.dense_e6(x))    # (b, 2304)

        mean = self.dense_mean(x)       # (b, 1728)
        logvar = self.dense_logvar(x)

        return mean, logvar, residuals

    def decode(self, z, residuals):
        # Pass reparameterized values z through dense layer and unflatten
        z = F.silu(self.dense_dec(z))     # (b, 2304)
        #z = F.silu(self.dense_d6(z))    # (b, 3072)
        #z = F.silu(self.dense_d5(z))    # (b, 4096)
        z = z.view(-1, 256, 4, 4)

        # Get just mask residual from last encoder layer
        # (256, 4, 4) -> (256, 4, 4)
        _, mask = residuals.pop()
        z, mask = self.d4(z, mask)

        # Concatenate z, mask w/ residuals, then upscale by 2
        # (256, 4, 4) -> (512, 4, 4) -> (512, 8, 8) -> (128, 8, 8)
        z_res, mask_res = residuals.pop()
        z = self.upscale(torch.concat((z, z_res), axis=1))
        mask = self.upscale(torch.concat((mask, mask_res), axis=1))
        z, mask = self.d3(z, mask)

        # Concatenate z, mask w/ residuals, then upscale by 2
        # (128, 8, 8) -> (256, 8, 8) -> (256, 16, 16) -> (64, 16, 16)
        z_res, mask_res = residuals.pop()
        z = self.upscale(torch.concat((z, z_res), axis=1))
        mask = self.upscale(torch.concat((mask, mask_res), axis=1))
        z, mask = self.d2(z, mask)

        # Concatenate z, mask w/ residuals, then upscale by 2
        # (64, 16, 16) -> (128, 16, 16) -> (128, 32, 32) -> (32, 32, 32)
        z_res, mask_res = residuals.pop()
        z = self.upscale(torch.concat((z, z_res), axis=1))
        mask = self.upscale(torch.concat((mask, mask_res), axis=1))
        z, mask = self.d1(z, mask)

        # Concatenate z, mask w/ residuals
        # (32, 32, 32) -> (64, 32, 32) -> (3, 32, 32)
        z_res, mask_res = residuals.pop()
        z = torch.concat((z, z_res), axis=1)
        mask = torch.concat((mask, mask_res), axis=1)
        z, _ = self.d0(z, mask)

        # Refeature (3, 32, 32) -> (1, 32, 32)
        return self.out_ref(z)
    
    def forward(self, x, mask):
        mean, logvar, residuals = self.encode(x, mask)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, residuals), mean, logvar