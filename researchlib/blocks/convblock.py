from torch import nn
from .basic_components import get_down_sampling_fn, get_up_sampling_fn, get_norm_fn
from ..models import builder


class _ConvBlock2d(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=3,
                 norm='batch',
                 activator=nn.ELU,
                 pooling=True,
                 pooling_type='combined',
                 pooling_factor=2,
                 preact=False,
                 se=False,
                 sn=False,
                 groups=1,
                 stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_dim,
                              out_dim,
                              kernel_size,
                              stride,
                              padding,
                              groups=groups,
                              bias=False)
        if sn: self.conv = nn.utils.spectral_norm(self.conv)
        bn_dim = in_dim if preact else out_dim
        self.bn = get_norm_fn(bn_dim, norm)
        self.activator = activator()
        self.pooling = pooling
        self.preact = preact
        if pooling:
            self.pooling_f = get_down_sampling_fn(out_dim, out_dim,
                                                  pooling_factor, preact,
                                                  pooling_type)

    def forward(self, x):
        if self.preact:
            x = self.bn(x)
            x = self.activator(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activator(x)
        if self.pooling: x = self.pooling_f(x)
        return x


class _ConvTransposeBlock2d(_ConvBlock2d):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=3,
                 norm='batch',
                 activator=nn.ELU,
                 pooling=True,
                 pooling_type='interpolate',
                 pooling_factor=2,
                 preact=False,
                 se=False,
                 sn=False,
                 groups=1,
                 stride=1):
        super().__init__(in_dim, out_dim, kernel_size, norm, activator,
                         pooling, pooling_type, pooling_factor, preact, se, sn,
                         groups, stride)
        if pooling:
            self.pooling_f = get_up_sampling_fn(out_dim, pooling_factor,
                                                preact, pooling_type)
