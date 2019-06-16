from torch import nn
import torch.nn.functional as F
from ..layers import *
from ..wrapper import *
from .builder import builder        

def _get_op_type(type):
    if type =='residual': 
        _op_type = block.ResBlock2d
        _op_transpose_type = block.ResTransposeBlock2d
    elif type =='resnext':
        _op_type = block.ResNextBlock2d
        _op_transpose_type = block.ResNextTransposeBlock2d
    elif type =='vgg':
        _op_type = block.ConvBlock2d
        _op_transpose_type = block.ConvTransposeBlock2d
    elif type =='dense':
        _op_type = block.DenseBlock2d
        _op_transpose_type = block.DenseTransposeBlock2d
    return _op_type, _op_transpose_type


def AutoUNet2d(input_dim, 
                    blocks,
                    down_type='residual',
                    up_type='vgg', 
                    start_filter=128, 
                    max_filter=1024,
                    down_pooling_type='maxpool',
                    up_pooling_type='convtranspose',
                    pooling_factor=2, 
                    pooling_freq=1, 
                    norm='batch', 
                    activator=nn.ELU, 
                    flatten=False, 
                    preact=True, 
                    attention=False,
                    input_multiscale=False,
                    return_multiscale=False,
                    se=True):


    _op_type, _op_transpose_type = _get_op_type(down_type)
    
    size_queue = []
    
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
            size_queue.append(out_dim)
            if attention:
                layers.append(
                    MultiscaleOutput(block.AttentionBlock2d(_op_type, _op_transpose_type, in_dim, out_dim, norm=norm, activator=activator, pooling_type=down_pooling_type, pooling_factor=pooling_factor, preact=preact, se=se)
                    ))
            else:
                layers.append(
                    MultiscaleOutput(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling_type=down_pooling_type, pooling_factor=pooling_factor, preact=preact, se=se)
                    ))
            count = 0
            in_dim = out_dim
            if out_dim < max_filter:
                out_dim *= 2
        else:
            if attention:
                layers.append(block.AttentionBlock2d(_op_type, _op_transpose_type, in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            else:
                layers.append(_op_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            in_dim = out_dim
    
    # Body
    print(in_dim, in_dim)
    if attention:
        layers.append(block.AttentionBlock2d(_op_type, _op_transpose_type, in_dim, in_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
    else:
        layers.append(_op_type(in_dim, in_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
    
    downpath = builder(layers)
    
    
    # Up path
    _op_type, _op_transpose_type = _get_op_type(up_type)
    
    layers = []
    count = 0
    for i in range(blocks):
        count += 1
        if count == pooling_freq:
            concat_in_dim = in_dim + size_queue.pop()
            print(concat_in_dim, out_dim)
            if attention:
                layers.append(
                    MultiscaleInput(block.AttentionTransposeBlock2d(_op_type, _op_transpose_type, concat_in_dim, out_dim, norm=norm, activator=activator, pooling_type=up_pooling_type, pooling_factor=pooling_factor, preact=preact, se=se)
                    ))
            else:
                layers.append(
                    MultiscaleInput(_op_transpose_type(concat_in_dim, out_dim, norm=norm, activator=activator, pooling_type=up_pooling_type, pooling_factor=pooling_factor, preact=preact, se=se)
                    ))
            count = 0
            in_dim = out_dim
            if out_dim > start_filter:
                out_dim = int(out_dim / 2)
        else:
            print(in_dim, out_dim)
            if attention:
                layers.append(block.AttentionTransposeBlock2d(_op_type, _op_transpose_type, in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            else:
                layers.append(_op_transpose_type(in_dim, out_dim, norm=norm, activator=activator, pooling=False, preact=preact, se=se))
            in_dim = out_dim
    
    uppath = builder(layers)
    
    return builder([downpath, uppath])
    