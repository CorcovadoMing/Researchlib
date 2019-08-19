import re
import math
from ..runner import Runner
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


def _filter_policy(base_dim, block_group, cur_dim, total_blocks, policy):
    if policy == 'default':
        return base_dim * (2**(block_group - 1))
    elif policy == 'pyramid':
        N = (total_blocks / 2) if total_blocks < 16 else (total_blocks / 3)
        return math.floor(cur_dim + 200 / N)


def AutoConvNet(op,
                input_dim,
                total_blocks,
                type='vgg',
                filters=(128, 1024),
                filter_policy='default',
                flatten=False,
                preact=True,
                pool_freq=1,
                do_norm=True,
                **kwargs):

    Runner.__model_settings__[
        f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    _op_type = _get_op_type(type)
    base_dim, max_dim = filters
    block_group = 1

    layers = []

    in_dim = input_dim
    out_dim = base_dim

    if preact:
        print(in_dim, out_dim)
        layers.append(layer.__dict__['Conv' + _get_dim_type(op)](
            in_dim, out_dim, 3, 1,
            1))  # Preact first layer is simply a hardcore transform
        in_dim = out_dim

    for i in range(total_blocks):
        id = i + 1

        if id % pool_freq == 0:
            do_pool = True
            block_group += 1
        else:
            do_pool = False

        if not preact and id == 1:
            out_dim = base_dim
        else:
            out_dim = _filter_policy(base_dim, block_group, in_dim,
                                     total_blocks, filter_policy)
        print(in_dim, out_dim)

        layers.append(
            _op_type(
                op,
                in_dim,
                out_dim,
                do_pool=do_pool,
                do_norm=do_norm,
                preact=preact,
                id=id,
                total_blocks=total_blocks,
                **kwargs))

        in_dim = out_dim

    if flatten:
        layers.append(layer.Flatten())
    return builder(layers)
