from torch import nn
import torch.nn.functional as F


class _FuseSampling2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.transform = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        return self.transform(F.interpolate(x, scale_factor=self.scale_factor, recompute_scale_factor=True))