import torch
import torch.nn.functional as F
from torch import nn
from ...models import builder
from .basic_components import _CombinedDownSampling, _MaxPoolDownSampling, _InterpolateUpSampling, _ConvTransposeUpSampling

class _Mask(nn.Module):
    def __init__(self, down_unit, up_unit, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling_factor=2, preact=True, se=False):
        super().__init__()
        self.down_unit = down_unit(in_dim, out_dim, norm, activator, True, 'maxpool', pooling_factor, preact, se)
        self.up_unit = up_unit(out_dim, out_dim, norm, activator, True, 'interpolate', pooling_factor, preact, se)
        self.red = nn.Conv2d(in_dim, out_dim, 1)
        self.attention_f = nn.Sigmoid()
        
    def forward(self, x):
        out = self.down_unit(x)
        out = self.up_unit(out)
        out = out + self.red(x)
        return self.attention_f(out)


class _AttentionBlock2d(nn.Module):
    def __init__(self, down_unit, up_unit, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='combined', pooling_factor=2, preact=True, se=False):
        super().__init__()
        self.trunk_branch = builder([
            down_unit(in_dim, out_dim, norm, activator, False, pooling_type, pooling_factor, preact, se),
            down_unit(out_dim, out_dim, norm, activator, False, pooling_type, pooling_factor, preact, se)
        ])
        self.mask_branch = _Mask(down_unit, up_unit, in_dim, out_dim, norm, activator, pooling_factor, preact, se)
        self.pooling = pooling
        if pooling: 
            if pooling_type == 'combined':
                self.pooling_f = _CombinedDownSampling(out_dim, pooling_factor, preact)
            elif pooling_type == 'maxpool':
                self.pooling_f = _MaxPoolDownSampling(out_dim, pooling_factor, preact)
        
    def forward(self, x):
        mask = self.mask_branch(x)
        trunk = self.trunk_branch(x)
        x = (1 + mask) * trunk
        if self.pooling: x = self.pooling_f(x)
        return x


class _AttentionTransposeBlock2d(_AttentionBlock2d):
    def __init__(self, down_unit, up_unit, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='interpolate', pooling_factor=2, preact=True, se=False):
        super().__init__(down_unit, up_unit, in_dim, out_dim, norm, activator, pooling, pooling_type, pooling_factor, preact, se)
        if pooling: 
            if pooling_type == 'interpolate':
                self.pooling_f = _InterpolateUpSampling(out_dim, pooling_factor, preact)
            elif pooling_type == 'convtranspose':
                self.pooling_f = _ConvTransposeUpSampling(out_dim, pooling_factor, preact)
        