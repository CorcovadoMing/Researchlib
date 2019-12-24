from torch import nn
import torch


class _PixelShuffle2d(nn.Module):
    def __init__(self, scale_factor, out_dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(out_dim, out_dim, 4, 2, 1)
        self.ups = nn.Upsample(None, scale_factor)
        self.ps = nn.PixelShuffle(scale_factor)
        hidden_channels = int(out_dim/(scale_factor**2))
        self.reduce = nn.Conv2d(hidden_channels+2*out_dim, out_dim, 1)
        
    
    def forward(self, x):
        x1 = self.ups(x) 
        x2 = self.ps(x)
        x3 = self.conv(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.reduce(x)