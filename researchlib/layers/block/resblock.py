from torch import nn
from .basic_components import _DownSampling, _UpSampling
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from ...models import builder

class _ResBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2):
        super().__init__()
        self.branch = builder([
            _ConvBlock2d(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=True),
            _ConvBlock2d(out_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=True)
        ])
        self.pooling = pooling
        if pooling: self.pooling_f = _DownSampling(out_dim, pooling_factor)
        self.activator = activator()
        self.red = True if in_dim != out_dim else False
        self.red_f = nn.Conv2d(in_dim, out_dim, 1)
        
    def forward(self, x):
        x_ = self.branch(x)
        if self.red: x = self.red_f(x)
        x = x + x_
        x = self.activator(x)
        if self.pooling: x = self.pooling_f(x)
        return x


class _ResTransposeBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2):
        super().__init__()
        self.branch = builder([
            _ConvBlock2d(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=True),
            _ConvBlock2d(out_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=True)
        ])
        self.pooling = pooling
        if pooling: self.pooling_f = builder([_UpSampling(),
                                              nn.Conv2d(out_dim, out_dim, 3, 1, 1)])
        self.activator = activator()
        self.red = True if in_dim != out_dim else False
        self.red_f = nn.Conv2d(in_dim, out_dim, 1)
        
    def forward(self, x):
        x_ = self.branch(x)
        if self.red: x = self.red_f(x)
        x = x + x_
        x = self.activator(x)
        if self.pooling: x = self.pooling_f(x)
        return x