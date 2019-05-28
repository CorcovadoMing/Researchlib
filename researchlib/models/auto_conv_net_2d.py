from torch import nn
import torch.nn.functional as F
from ..layers import *
from .builder import builder        

def AutoConvNet2d(input_dim, blocks, start_filter=128, max_filter=1024, pooling_factor=2, pooling_freq=1, norm='batch', activator=nn.ELU, flatten=False, residual=False):
    _op_type = block.ResBlock2d if residual else block.ConvBlock2d
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    for i in range(blocks):
        print(in_dim, out_dim)
        count += 1
        if count == pooling_freq:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling_factor=pooling_factor))
            count = 0
            in_dim = out_dim
            if out_dim < max_filter:
                out_dim *= 2
        else:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False))
            in_dim = out_dim
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
        
    
def AutoConvTransposeNet2d(input_dim, blocks, start_filter=1024, min_filter=128, pooling_factor=2, pooling_freq=1, norm='batch', activator=nn.ELU, flatten=False, residual=False):
    _op_type = block.ResTransposeBlock2d if residual else block.ConvTransposeBlock2d
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    for i in range(blocks):
        print(in_dim, out_dim)
        count += 1
        if count == pooling_freq:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling_factor=pooling_factor))
            count = 0
            in_dim = out_dim
            if out_dim > min_filter:
                out_dim = int(out_dim / 2)
        else:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False))
            in_dim = out_dim
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)