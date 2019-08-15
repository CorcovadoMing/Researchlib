import re
from ..layers import layer
from ..blocks import block
from .builder import builder

from ..blocks._convblock import ConvBlock as cb
from ..blocks._resblock import ResBlock as rb

# =============================================================


def _get_param(kwargs, key, init_value):
    try:
        query = kwargs[key]
        return query
    except:
        return init_value


def _get_op_type(type):
    if type == 'vgg':
        _op_type = cb
    elif type == 'residual':
        _op_type = rb
    return _op_type


def _get_dim_type(op):
    match = re.search('\dd', str(op))
    dim_str = match.group(0)
    return dim_str


# =============================================================


def AutoConvNet(op,
                input_dim,
                total_blocks,
                type='vgg',
                filters=(128, 1024),
                flatten=False,
                preact=True,
                pool_freq=1,
                do_norm=True,
                **kwargs):

    _op_type = _get_op_type(type)
    start_filter, max_filter = filters

    layers = []

    in_dim = input_dim
    out_dim = start_filter

    print(in_dim, out_dim)

    if preact:
        layers.append(layer.__dict__['Conv' + _get_dim_type(op)](
            in_dim, out_dim, 3, 1,
            1))  # Preact first layer is simply a hardcore transform

    for i in range(total_blocks):
        id = i + 1
        in_dim = out_dim
        if id % pool_freq == 0:
            do_pool = True
            if out_dim < max_filter:
                out_dim *= 2
        else:
            do_pool = False

        layers.append(
            _op_type(op,
                     in_dim,
                     out_dim,
                     do_pool=do_pool,
                     do_norm=do_norm,
                     preact=preact,
                     id=id,
                     total_blocks=total_blocks,
                     **kwargs)
        )

        print(in_dim, out_dim)
    if flatten: layers.append(layer.Flatten())
    return builder(layers)
