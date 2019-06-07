import torch
import torch.nn.functional as F
from torch import nn
from .basic_components import _DownSampling, _UpSampling
from ...models import builder

class _ResNextBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2, preact=True, se=False):
        super().__init__()
        groups = min(out_dim, 32)
        if norm =='batch': bn = nn.BatchNorm2d
        elif norm == 'instance': bn = nn.InstanceNorm2d
        if preact:
            self.branch = builder([
                bn(in_dim),
                activator(),
                nn.Conv2d(in_dim, out_dim, 1),
                bn(out_dim),
                activator(),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=groups),
                bn(out_dim),
                activator(),
                nn.Conv2d(out_dim, out_dim, 1),
            ])
        else:
            self.branch = builder([
                nn.Conv2d(in_dim, out_dim, 1),
                bn(out_dim),
                activator(),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=groups),
                bn(out_dim),
                activator(),
                nn.Conv2d(out_dim, out_dim, 1),
                bn(out_dim),
                activator(),
            ])
        self.pooling = pooling
        if pooling: self.pooling_f = _DownSampling(out_dim, pooling_factor, preact)
        self.red = True if in_dim != out_dim else False
        self.red_f = nn.Conv2d(in_dim, out_dim, 1)
        
        self.se = se
        if se:
            self.fc1 = nn.Conv2d(out_dim, out_dim//16, kernel_size=1)
            self.fc2 = nn.Conv2d(out_dim//16, out_dim, kernel_size=1)
        
    def forward(self, x):
        x_ = self.branch(x)
        
        if self.se:
            # Squeeze
            w = F.adaptive_avg_pool2d(x_, (1, 1))
            w = F.relu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))
            # Excitation
            x_ = x_ * w
        
        if self.red: x = self.red_f(x)
        x = x + x_
        if self.pooling: x = self.pooling_f(x)
        return x


class _ResNextTransposeBlock2d(_ResNextBlock2d):
    def __init__(self, in_dim, out_dim, norm='batch', activator=nn.ELU, pooling=True, pooling_factor=2, preact=True, se=False):
        super().__init__(in_dim, out_dim, norm, activator, pooling, pooling_factor, preact, se)
        if pooling: self.pooling_f = builder([_UpSampling(out_dim, pooling_factor, preact),
                                              nn.Conv2d(out_dim, out_dim, 3, 1, 1)])
