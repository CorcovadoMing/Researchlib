import re
import math
from ..runner import Runner
from ..layers import layer
from ..blocks import block
from .builder import builder

from ..blocks._resblock import ResBlock as rb
from ..blocks._resblock_bottleneck import ResBottleneckBlock as rbb
from ..blocks._wide_resblock import WideResBlock as wrb

# =============================================================


def _get_param(kwargs, key, init_value):
    try:
        query = kwargs[key]
        return query
    except:
        return init_value


def _get_op_type(type):
    if type not in ['residual', 'residual-bottleneck', 'wide-residual']:
        raise ('Type is not supperted')
    if type == 'residual':
        _op_type = rb
    elif type == 'residual-bottleneck':
        _op_type = rbb
    elif type == 'wide-residual':
        _op_type = wrb
    return _op_type


def _get_dim_type(op):
    match = re.search('\dd', str(op))
    dim_str = match.group(0)
    return dim_str


# =============================================================


def _filter_policy(base_dim, block_group, cur_dim, total_blocks, policy,
                   kwargs):
    if policy == 'default':
        return base_dim * (2**(block_group - 1))
    elif policy == 'pyramid':
        pyramid_alpha = _get_param(kwargs, 'pyramid_alpha', 48)
        N = (total_blocks / 2) if total_blocks < 16 else (total_blocks / 3)
        return math.floor(cur_dim + pyramid_alpha / N)


def AutoConvNet(
        op,
        unit,
        input_dim,
        total_blocks,
        type='residual',
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

    wide_scale = _get_param(
        kwargs, 'wide_scale',
        10) if type == 'wide-residual' and filter_policy != 'pyramid' else 1
    in_dim = input_dim
    out_dim = wide_scale * base_dim

    if preact:
        print(in_dim, out_dim)
        layers.append(layer.__dict__['Conv' + _get_dim_type(op)](
            in_dim, out_dim, 3, 1,
            1))  # Preact first layer is simply a hardcore transform
        in_dim = out_dim

    for i in range(total_blocks):
        id = i + 1

        if not preact and id == 1:
            out_dim = wide_scale * base_dim
        else:
            out_dim = wide_scale * _filter_policy(base_dim, block_group, in_dim,
                                                  total_blocks, filter_policy,
                                                  kwargs)

        if id % pool_freq == 0:
            do_pool = True
            block_group += 1
        else:
            do_pool = False

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
                #                 unit=unit,
                **kwargs))

        in_dim = out_dim

    if flatten:
        layers.append(layer.Flatten())

    # must verify after all keys get registered
    block.Block.verify_kwargs(**kwargs)

    return builder(layers)
