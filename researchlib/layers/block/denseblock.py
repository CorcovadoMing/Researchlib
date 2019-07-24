import torch
import torch.nn.functional as F
from torch import nn
from .basic_components import get_down_sampling_fn, get_up_sampling_fn
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from ...models import builder


class _DenseBlock2d(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='batch',
                 activator=nn.ELU,
                 pooling=True,
                 pooling_type='combined',
                 pooling_factor=2,
                 preact=True,
                 se=False):
        super().__init__()
        k = 24
        self.branch = []
        inner_in = in_dim
        inner_out = k
        for i in range(6):
            self.branch.append(
                builder([
                    _ConvBlock2d(inner_in,
                                 inner_out,
                                 kernel_size=1,
                                 norm=norm,
                                 activator=activator,
                                 pooling=False,
                                 preact=preact),
                    _ConvBlock2d(inner_out,
                                 inner_out,
                                 kernel_size=3,
                                 norm=norm,
                                 activator=activator,
                                 pooling=False,
                                 preact=preact)
                ]))
            inner_in += inner_out
        self.branch = nn.ModuleList(self.branch)
        self.pooling = pooling
        if pooling:
            self.pooling_f = get_down_sampling_fn(out_dim, pooling_factor,
                                                  preact, pooling_type)


#         self.se = se
#         if se:
#             self.fc1 = nn.Conv2d(out_dim, out_dim//16, kernel_size=1)
#             self.fc2 = nn.Conv2d(out_dim//16, out_dim, kernel_size=1)

    def forward(self, x):
        global_feature = x
        for i in range(6):
            x = self.branch[i](global_feature)
            global_feature = torch.cat([global_feature, x], dim=1)
        x = self.transition(global_feature)
        if self.pooling: x = self.pooling_f(x)
        return x


class _DenseTransposeBlock2d(_DenseBlock2d):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='batch',
                 activator=nn.ELU,
                 pooling=True,
                 pooling_type='interpolate',
                 pooling_factor=2,
                 preact=True,
                 se=False):
        super().__init__(in_dim, out_dim, norm, activator, pooling,
                         pooling_type, pooling_factor, preact, se)
        if pooling:
            self.pooling_f = get_up_sampling_fn(out_dim, pooling_factor,
                                                preact, pooling_type)
