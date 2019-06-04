from torch import nn
import torch.nn.functional as F
from ..layers import *
from .builder import builder        

def AutoConvNet2d(input_dim, blocks, type='vgg', start_filter=128, max_filter=1024, pooling_factor=2, pooling_freq=1, norm='batch', activator=nn.ELU, flatten=False, preact=True, attention=False):
    if type =='residual': _op_type = block.ResBlock2d
    elif type =='resnext': _op_type = block.ResNextBlock2d
    elif type =='vgg': _op_type = block.ConvBlock2d
    elif type =='dense': _op_type = block.DenseBlock2d
    se = True
    
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    
    if preact: 
        print(in_dim, out_dim)
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        in_dim = out_dim
    
    for i in range(blocks):
        print(in_dim, out_dim)
        count += 1
        if count == pooling_freq:
            if attention:
                layers.append(block.AttentionBlock(block.ResNextBlock2d, block.ResNextTransposeBlock2d, in_dim, out_dim, norm=norm, activator=activator, pooling_factor=pooling_factor, preact=preact, se=se))
            else:
                layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling_factor=pooling_factor, preact=preact, se=se))
            count = 0
            in_dim = out_dim
            if out_dim < max_filter:
                out_dim *= 2
        else:
            if attention:
                layers.append(block.AttentionBlock(block.ResNextBlock2d, block.ResNextTransposeBlock2d, in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            else:
                layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            in_dim = out_dim
    if flatten: layers.append(layer.Flatten())
    return builder(layers)
        
    
def AutoConvTransposeNet2d(input_dim, blocks, type='vgg', start_filter=1024, min_filter=128, pooling_factor=2, pooling_freq=1, norm='batch', activator=nn.ELU, flatten=False, preact=True):
    if type =='residual': _op_type = block.ResTransposeBlock2d
    elif type =='resnext': _op_type = block.ResNextTransposeBlock2d
    elif type =='vgg': _op_type = block.ConvTransposeBlock2d
    elif type =='dense': _op_type = block.DenseTransposeBlock2d
    se = True
    
    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0
    
    if preact: 
        print(in_dim, out_dim)
        layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        in_dim = out_dim
    
    for i in range(blocks):
        print(in_dim, out_dim)
        count += 1
        if count == pooling_freq:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling_factor=pooling_factor, preact=preact, se=se))
            count = 0
            in_dim = out_dim
            if out_dim > min_filter:
                out_dim = int(out_dim / 2)
        else:
            layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            in_dim = out_dim
    if flatten: layers.append(layer.Flatten())
    return builder(layers)