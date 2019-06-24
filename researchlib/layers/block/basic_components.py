import torch
from torch import nn
import torch.nn.functional as F

def get_down_sampling_fn(in_dim, out_dim, pooling_factor, preact, pooling_type):
    pooling_f = None
    if pooling_type == 'combined':
        pooling_f = _CombinedDownSampling(in_dim, out_dim, pooling_factor, preact)
    elif pooling_type == 'maxpool':
        pooling_f = _MaxPoolDownSampling(in_dim, out_dim, pooling_factor, preact)
    elif pooling_type == 'avgpool':
        pooling_f = _AvgPoolDownSampling(in_dim, out_dim, pooling_factor, preact)
    elif pooling_type == 'k3stride':
        pooling_f = _Convk3StrideDownSampling(in_dim, out_dim, pooling_factor, preact)
    elif pooling_type == 'k1stride':
        pooling_f = _Convk1StrideDownSampling(in_dim, out_dim, pooling_factor, preact)
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
    
def get_norm_fn(bn_dim, norm):
    if norm =='batch': 
        return nn.BatchNorm2d(bn_dim)
    elif norm == 'instance': 
        return nn.GroupNorm(bn_dim, bn_dim)
    elif norm == 'group': 
        return nn.GroupNorm(int(bn_dim/4), bn_dim)
    elif norm == 'layer': 
        return nn.GroupNorm(1, bn_dim)

# -----------------------------------------------------------------------

class _MaxPoolDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
        
    def forward(self, x):
        return self.m(x)


class _AvgPoolDownSampling(_MaxPoolDownSampling):
    def __init__(self, in_dim, out_dim, pooling_factor, preact=False):
        super().__init__(in_dim, pooling_factor, preact)
        self.m = nn.AvgPool2d(pooling_factor)


class _Convk3StrideDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, pooling_factor, preact=False):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, pooling_factor, 1, bias=False)
        self.preact = preact
        if preact:
            self.bn = nn.BatchNorm2d(in_dim)
        else:
            self.bn = nn.BatchNorm2d(out_dim)
        self.activator = nn.ReLU()
        
    def forward(self, x):
        if self.preact:
            x = self.bn(x)
            x = self.activator(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activator(x)
        return x


class _Convk1StrideDownSampling(_Convk3StrideDownSampling):
    def __init__(self, in_dim, out_dim, pooling_factor, preact=False):
        super().__init__(in_dim, out_dim, pooling_factor, preact)
        self.conv = nn.Conv2d(in_dim, out_dim, 1, pooling_factor, bias=False)


class _CombinedDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = _MaxPoolDownSampling(in_dim, in_dim, pooling_factor, preact)
        self.a = _AvgPoolDownSampling(in_dim, in_dim, pooling_factor, preact)
        self.c = _Convk3StrideDownSampling(in_dim, in_dim, pooling_factor, preact)
        self.red = nn.Conv2d(in_dim*3, out_dim, 1)
        self.activator = nn.ReLU()
        self.preact = preact
        if preact:
            self.bn = nn.BatchNorm2d(in_dim*3)
        else:
            self.bn = nn.BatchNorm2d(out_dim)
    
    def forward(self, x):
        x = torch.cat([self.m(x), self.a(x), self.c(x)], dim=1)
        if self.preact:
            x = self.bn(x)
            x = self.activator(x)
            x = self.red(x)
        else:
            x = self.red(x)
            x = self.bn(x)
            x = self.activator(x)
        return x

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
        if preact:
            self.bn = nn.BatchNorm2d(int(in_dim/(pooling_factor**2)))
        else:
            self.bn = nn.BatchNorm2d(in_dim)
        self.activator = nn.ELU()
        self.preact = preact
        
    def forward(self, x):
        x = self.m(x)
        if self.preact: 
            x = self.bn(x)
            x = self.activator(x)
            x = self.red(x)
        else:
            x = self.red(x)
            x = self.bn(x)
            x = self.activator(x)
        return x
        