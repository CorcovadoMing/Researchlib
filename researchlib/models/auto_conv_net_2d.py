from torch import nn
import torch.nn.functional as F
from ..layers import *
from .builder import builder

class _DownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
        self.a = nn.AvgPool2d(pooling_factor)
        self.c = builder([
                    nn.Conv2d(in_dim, in_dim, 3, pooling_factor, 1),
                    nn.LeakyReLU(0.2)
                    ])
        self.red = builder([
                    nn.Conv2d(in_dim*3, in_dim, 1),
                    nn.LeakyReLU(0.2)
                    ])
                    
    
    def forward(self, x):
        x = torch.cat([self.m(x), self.a(x), self.c(x)], dim=1)
        return self.red(x)
        


def AutoConvNet2d(input_dim, blocks, start_filter=128, max_filter=1024, pooling_factor=2, pooling_freq=1, bn=True, activator=nn.ELU, flatten=False):
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    for i in range(blocks):
        print(in_dim, out_dim)
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        if bn:
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(activator())
        in_dim = out_dim
        count += 1
        if count == pooling_freq:
            layers.append(_DownSampling(in_dim, pooling_factor))
            count = 0
            if out_dim < max_filter:
                out_dim *= 2
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
    
    
class _UpSampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.activator = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.activator(x)
    
    
def AutoConvTransposeNet2d(input_dim, blocks, start_filter=1024, min_filter=128, pooling_factor=2, pooling_freq=1, bn=True, activator=nn.ELU, flatten=False):
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    for i in range(blocks):
        print(in_dim, out_dim)
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        if bn:
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(activator())
        in_dim = out_dim
        count += 1
        if count == pooling_freq:
            layers.append(_UpSampling())
            layers.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
            count = 0
            if out_dim > min_filter:
                out_dim /= 2
                out_dim = int(out_dim)
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)