import torch
import torch.nn.functional as F
from torch import nn
from ...models import builder
from .basic_components import _DownSampling, _UpSampling

class _AttentionBlock2d(nn.Module):
    def __init__(self, down_unit, up_unit, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2, preact=True, se=False):
        super().__init__()
        self.trunk_branch = builder([
            down_unit(in_dim, out_dim, norm, activator, False, pooling_factor, preact, se),
            down_unit(out_dim, out_dim, norm, activator, False, pooling_factor, preact, se)
        ])
        self.mask_branch = builder([
            down_unit(in_dim, out_dim, norm, activator, True, pooling_factor, preact, se),
            up_unit(out_dim, out_dim, norm, activator, True, pooling_factor, preact, se),
            nn.Sigmoid()
        ])
        self.pooling = pooling
        if pooling: self.pooling_f = _DownSampling(out_dim, pooling_factor, preact)
        
    def forward(self, x):
        mask = self.mask_branch(x)
        trunk = self.trunk_branch(x)
        x = (1 + mask) * trunk
        if self.pooling: x = self.pooling_f(x)
        return x

class _AttentionTransposeBlock2d(_AttentionBlock2d):
    def __init__(self, down_unit, up_unit, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2, preact=True, se=False):
        super().__init__(down_unit, up_unit, in_dim, out_dim, norm, activator, pooling, pooling_factor, preact, se)
        if pooling: self.pooling_f = builder([_UpSampling(out_dim, pooling_factor, preact),
                                              nn.Conv2d(out_dim, out_dim, 3, 1, 1)])