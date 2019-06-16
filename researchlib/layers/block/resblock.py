import torch
import torch.nn.functional as F
from torch import nn
from .basic_components import get_down_sampling_fn, get_up_sampling_fn
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from ...models import builder

class _ResBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='combined', pooling_factor=2, preact=True, se=False):
        super().__init__()
        
        self.pooling = pooling
        if self.pooling: 
            _pooling_factor = pooling_factor
            self.shortcut_reduce = _ConvBlock2d(in_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact, stride=_pooling_factor)
        else:
            _pooling_factor = 1
            
        self.branch = builder([
            _ConvBlock2d(in_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact, stride=_pooling_factor),
            _ConvBlock2d(out_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact)
        ])
        
        self.se = se
        if se:
            self.se_branch = builder([
                nn.AdaptiveMaxPool2d(1),
                nn.Conv2d(out_dim, out_dim//16, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_dim//16, out_dim, kernel_size=1),
                nn.Sigmoid()
            ])
        
    def forward(self, x):
        x_ = self.branch(x)
        if self.se: x_ = x_ * self.se_branch(x_)
        if self.pooling: x = self.shortcut_reduce(x)
        return x + x_


class _ResTransposeBlock2d(_ResBlock2d):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='interpolate', pooling_factor=2, preact=True, se=False):
        super().__init__(in_dim, out_dim, norm, activator, pooling, pooling_type, pooling_factor, preact, se)
        if pooling: self.pooling_f = get_up_sampling_fn(out_dim, pooling_factor, preact, pooling_type)