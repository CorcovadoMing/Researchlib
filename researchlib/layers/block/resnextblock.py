import torch
import torch.nn.functional as F
from torch import nn
from .basic_components import get_down_sampling_fn, get_up_sampling_fn
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from ...models import builder
from .resblock import _ResBlock2d

class _ResNextBlock2d(_ResBlock2d):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='combined', pooling_factor=2, preact=True, se=False):
        super().__init__(in_dim, out_dim, norm, activator, pooling, pooling_type, pooling_factor, preact, se)
        groups = max(int(out_dim/4), 64)
        self.pooling = pooling
        if self.pooling: _pooling_factor = pooling_factor
        else: _pooling_factor = 1
        self.branch = builder([
            _ConvBlock2d(in_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
            _ConvBlock2d(out_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact, stride=_pooling_factor, groups=groups),
            _ConvBlock2d(out_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
            _ConvBlock2d(out_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact, groups=groups),
            _ConvBlock2d(out_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
        ])

# class _ResNextBlock2d(nn.Module):
#     def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='combined', pooling_factor=2, preact=True, se=False):
#         super().__init__()
#         groups = min(out_dim, 64)
#         self.branch = builder([
#             _ConvBlock2d(in_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
#             _ConvBlock2d(out_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact, groups=groups),
#             _ConvBlock2d(out_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
#             _ConvBlock2d(out_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
#             _ConvBlock2d(out_dim, out_dim, kernel_size=3, norm=norm, activator=activator, pooling=False, preact=preact, groups=groups),
#             _ConvBlock2d(out_dim, out_dim, kernel_size=1, norm=norm, activator=activator, pooling=False, preact=preact),
#         ])
#         self.pooling = pooling
#         if pooling: self.pooling_f = get_down_sampling_fn(out_dim, pooling_factor, preact, pooling_type)
#         self.red = True if in_dim != out_dim else False
#         self.red_f = nn.Conv2d(in_dim, out_dim, 1)
        
#         self.se = se
#         if se:
#             self.fc1 = nn.Conv2d(out_dim, out_dim//16, kernel_size=1)
#             self.fc2 = nn.Conv2d(out_dim//16, out_dim, kernel_size=1)
        
#     def forward(self, x):
#         x_ = self.branch(x)
        
#         if self.se:
#             # Squeeze
#             w = F.adaptive_avg_pool2d(x_, (1, 1))
#             w = F.relu(self.fc1(w))
#             w = torch.sigmoid(self.fc2(w))
#             # Excitation
#             x_ = x_ * w
        
#         if self.red: x = self.red_f(x)
#         x = x + x_
#         if self.pooling: x = self.pooling_f(x)
#         return x


class _ResNextTransposeBlock2d(_ResNextBlock2d):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_type='interpolate', pooling_factor=2, preact=True, se=False):
        super().__init__(in_dim, out_dim, norm, activator, pooling, pooling_type, pooling_factor, preact, se)
        if pooling: self.pooling_f = get_up_sampling_fn(out_dim, pooling_factor, preact, pooling_type)
        