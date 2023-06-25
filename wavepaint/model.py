

import torch
import torch.nn as nn
import wavemix
from wavemix import Level1Waveblock
    

class WaveMixModule(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 2, 1),
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))

        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(final_dim*2, int(final_dim/2), 4, stride = 2, padding = 1),
            nn.BatchNorm2d(int(final_dim/2))
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(int(final_dim/2) + 3, 3, 1),
        )
        

    def forward(self, img, mask):

        x = torch.cat([img, mask], dim=1)

        x = self.conv(x)  

        skip1 = x

        for attn in self.layers:
            x = attn(x) + x

        x = self.depthconv(x)

        x = torch.cat([x, skip1], dim=1)  # skip connection

        x = self.decoder1(x)

        x = torch.cat([x, img], dim=1)  # skip connection

        x = self.decoder2(x)

        return x


class WavePaint(nn.Module):
    def __init__(
        self,
        *,
        num_modules= 1,
        blocks_per_module = 7,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.,
        
    ):
        super().__init__()
        
        self.wavemodules = nn.ModuleList([])
        for _ in range(num_modules):
            self.wavemodules.append(WaveMixModule(depth = blocks_per_module, mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))

        
    def forward(self, img, mask):
        x = img

        for module in self.wavemodules:
            x = module(x, mask) + x
            
        x = x*mask + img*(1-mask)

        return x
    
	


		




