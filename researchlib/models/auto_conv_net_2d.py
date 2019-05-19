from torch import nn
from ..layers import layer
from .builder import builder

def AutoConvNet2d(input_dim, blocks, start_filter=64, pooling_factor=2, pooling_freq=1, activator=nn.ELU, flatten=True):
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    for i in range(blocks):
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(activator())
        in_dim = out_dim
        if i % pooling_freq == 0:
            layers.append(nn.MaxPool2d(pooling_factor))
        out_dim *= 2
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
    
def AutoConvTransposeNet2d(input_dim, blocks, start_filter=64, pooling_factor=2, pooling_freq=1, activator=nn.ELU, flatten=False):
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    for i in range(blocks):
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(activator())
        in_dim = out_dim
        if i % pooling_freq == 0:
            layers.append(nn.ConvTranspose2d(out_dim, out_dim, pooling_factor, pooling_factor))
        out_dim *= 2
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)