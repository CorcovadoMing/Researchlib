from torch import nn
from ..layers import *
from .builder import builder

def AutoConvNet2d(input_dim, blocks, start_filter=128, max_filter=1024, pooling_factor=2, pooling_freq=1, bn=True, activator=nn.ELU, flatten=True):
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
            layers.append(nn.MaxPool2d(pooling_factor))
            count = 0
            if out_dim < max_filter:
                out_dim *= 2
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
    
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
            layers.append(nn.ConvTranspose2d(out_dim, out_dim, pooling_factor, pooling_factor))
            count = 0
            if out_dim > min_filter:
                out_dim /= 2
                out_dim = int(out_dim)
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)