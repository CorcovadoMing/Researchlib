import torch
from torch import nn
import torch.nn.functional as F

def get_down_sampling_fn(out_dim, pooling_factor, preact, pooling_type):
    pooling_f = None
    if pooling_type == 'combined':
        pooling_f = _CombinedDownSampling(out_dim, pooling_factor, preact)
    elif pooling_type == 'maxpool':
        pooling_f = _MaxPoolDownSampling(out_dim, pooling_factor, preact)
    elif pooling_type == 'avgpool':
        pooling_f = _AvgPoolDownSampling(out_dim, pooling_factor, preact)
    elif pooling_type == 'stride':
        pooling_f = _ConvStrideDownSampling(out_dim, pooling_factor, preact)
    return pooling_f

def get_up_sampling_fn(out_dim, pooling_factor, preact, pooling_type):
    pooling_f = None
    if pooling_type == 'interpolate':
        pooling_f = _InterpolateUpSampling(out_dim, pooling_factor, preact)
    elif pooling_type == 'convtranspose':
        pooling_f = _ConvTransposeUpSampling(out_dim, pooling_factor, preact)
    elif pooling_type == 'pixelshuffle':
        pooling_f = _PixelShuffleUpSampling(out_dim, pooling_factor, preact)
    return pooling_f

# -----------------------------------------------------------------------

class _CombinedDownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
        self.a = nn.AvgPool2d(pooling_factor)
        self.c = nn.Conv2d(in_dim, in_dim, 3, pooling_factor, 1)
        self.red = nn.Conv2d(in_dim*3, in_dim, 1)
        self.activator = nn.ELU()
        self.preact = preact
    
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = torch.cat([self.m(x), self.a(x), self.c(x)], dim=1)
        x = self.activator(x)
        x = self.red(x)
        if not self.preact: x = self.activator(x)
        return x


class _MaxPoolDownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
    
    def forward(self, x):
        return self.m(x)


class _AvgPoolDownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.AvgPool2d(pooling_factor)
    
    def forward(self, x):
        return self.m(x)


class _ConvStrideDownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.c = nn.Conv2d(in_dim, in_dim, 3, pooling_factor, 1)
    
    def forward(self, x):
        return self.c(x)


# -------------------------------------------------------------------------------------------

class _InterpolateUpSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.activator = nn.ELU()
        self.preact = preact
        self.pooling_factor = pooling_factor
        
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = F.interpolate(x, scale_factor=self.pooling_factor)
        x = self.conv(x)
        if not self.preact: x = self.activator(x)
        return x


class _ConvTransposeUpSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.ConvTranspose2d(in_dim, in_dim, pooling_factor, pooling_factor)
        self.activator = nn.ELU()
        self.preact = preact
        
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = self.m(x)
        if not self.preact: x = self.activator(x)
        return x
        
        
class _PixelShuffleUpSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.PixelShuffle(pooling_factor)
        self.red = nn.Conv2d(int(in_dim/(pooling_factor**2)), in_dim, 1)
        self.activator = nn.ELU()
        self.preact = preact
        
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = self.m(x)
        x = self.red(x)
        if not self.preact: x = self.activator(x)
        return x
        