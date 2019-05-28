from torch import nn
from .basic_components import _DownSampling, _UpSampling
from ...models import builder

class _ConvBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        if norm =='batch': self.bn = nn.BatchNorm2d(out_dim)
        elif norm == 'instance': self.bn = nn.InstanceNorm2d(out_dim)
        self.activator = activator()
        self.pooling = pooling
        if pooling: self.pooling_f = _DownSampling(out_dim, pooling_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activator(x)
        if self.pooling: x = self.pooling_f(x)
        return x


class _ConvTransposeBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        if norm =='batch': self.bn = nn.BatchNorm2d(out_dim)
        elif norm == 'instance': self.bn = nn.InstanceNorm2d(out_dim)
        self.activator = activator()
        self.pooling = pooling
        if pooling: self.pooling_f = builder([_UpSampling(),
                                              nn.Conv2d(out_dim, out_dim, 3, 1, 1)])
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activator(x)
        if self.pooling: x = self.pooling_f(x)
        return x