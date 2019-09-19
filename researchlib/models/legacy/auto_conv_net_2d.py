from torch import nn
import torch.nn.functional as F
from ..layers import layer
from ..blocks import block
from .builder import builder


def _get_op_type(type):
    if type == 'residual':
        _op_type = block.ResBlock2d
        _op_transpose_type = block.ResTransposeBlock2d
    elif type == 'resnext':
        _op_type = block.ResNextBlock2d
        _op_transpose_type = block.ResNextTransposeBlock2d
    elif type == 'vgg':
        _op_type = block.ConvBlock2d
        _op_transpose_type = block.ConvTransposeBlock2d
    elif type == 'dense':
        _op_type = block.DenseBlock2d
        _op_transpose_type = block.DenseTransposeBlock2d
    return _op_type, _op_transpose_type


def AutoConvNet2d(
    input_dim,
    block_num,
    type = 'vgg',
    filters = (128, 1024),
    pooling_type = 'combined',
    pooling_factor = 2,
    pooling_freq = 1,
    norm = 'batch',
    activator = nn.ReLU,
    flatten = False,
    preact = True,
    attention = False,
    input_multiscale = False,
    return_multiscale = False,
    se = True,
    sn = False,
    **kwargs
):

    _op_type, _op_transpose_type = _get_op_type(type)
    start_filter, max_filter = filters

    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0

    print(in_dim, out_dim)
    layers.append(
        block.ConvBlock2d(
            in_dim, out_dim, pooling = False, activator = activator, se = se, sn = sn
        )
    )

    for i in range(block_num):
        count += 1
        in_dim = out_dim
        if count == pooling_freq:
            if out_dim < max_filter:
                out_dim *= 2

            if attention:
                layers.append(
                    block.AttentionBlock2d(
                        _op_type,
                        _op_transpose_type,
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling_type = pooling_type,
                        pooling_factor = pooling_factor,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    _op_type(
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling_type = pooling_type,
                        pooling_factor = pooling_factor,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            count = 0
        else:
            if attention:
                layers.append(
                    block.AttentionBlock2d(
                        _op_type,
                        _op_transpose_type,
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling = False,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    _op_type(
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling = False,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
        print(in_dim, out_dim)
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)


def AutoConvTransposeNet2d(
    input_dim,
    block_num,
    type = 'vgg',
    filters = (1024, 128),
    pooling_type = 'interpolate',
    pooling_factor = 2,
    pooling_freq = 1,
    norm = 'batch',
    activator = nn.ELU,
    flatten = False,
    preact = True,
    attention = False,
    input_multiscale = False,
    return_multiscale = False,
    se = True,
    sn = False,
    **kwargs
):

    _op_type, _op_transpose_type = _get_op_type(type)
    start_filter, min_filter = filters

    layers = []
    in_dim = input_dim
    out_dim = start_filter
    count = 0

    #     print(in_dim, out_dim)
    #     layers.append(block.ConvBlock2d(in_dim, out_dim, pooling=False, activator=activator, se=se, sn=sn))

    for i in range(block_num):
        print(in_dim, out_dim)
        count += 1
        if count == pooling_freq:
            if attention:
                layers.append(
                    block.AttentionTransposeBlock2d(
                        _op_type,
                        _op_transpose_type,
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling_type = pooling_type,
                        pooling_factor = pooling_factor,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    _op_transpose_type(
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling_type = pooling_type,
                        pooling_factor = pooling_factor,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            count = 0
            in_dim = out_dim
            if out_dim > min_filter:
                out_dim = int(out_dim / 2)
        else:
            if attention:
                layers.append(
                    block.AttentionTransposeBlock2d(
                        _op_type,
                        _op_transpose_type,
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling = False,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    _op_transpose_type(
                        in_dim,
                        out_dim,
                        norm = norm,
                        activator = activator,
                        pooling = False,
                        preact = preact,
                        se = se,
                        sn = sn,
                        block_idx = i + 1,
                        block_num = block_num,
                        **kwargs
                    )
                )
            in_dim = out_dim
    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
