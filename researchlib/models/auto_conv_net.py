import re
import math
from ..runner import Runner
from ..layers import layer
from .builder import builder
from ..utils import ParameterManager

from ..blocks._resblock import ResBlock as rb
from ..blocks._resblock_bottleneck import ResBottleneckBlock as rbb
from ..blocks._wide_resblock import WideResBlock as wrb
from ..blocks._vggblock import VGGBlock as vb
from ..blocks._inverted_bottleneck import InvertedBottleneckBlock as ibb
from ..blocks._inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE

# =============================================================


def _get_op_type(type, cur_block, total_blocks):
    if type not in ['vgg', 'residual', 'residual-bottleneck', 'wide-residual', 'inverted-bottleneck', 'inception']:
        raise ('Type is not supperted')
    if type == 'vgg':
        _op_type = vb
    elif type == 'residual':
        _op_type = rb
    elif type == 'residual-bottleneck':
        _op_type = rbb
    elif type == 'wide-residual':
        _op_type = wrb
    elif type == 'inverted-bottleneck':
        _op_type = ibb
    elif type == 'inception':
        if (cur_block / total_blocks) <= 0.2:
            _op_type = InceptionA
        elif (cur_block / total_blocks) <= 0.4:
            _op_type = InceptionB
        elif (cur_block / total_blocks) <= 0.6:
            _op_type = InceptionC
        elif (cur_block / total_blocks) <= 0.8:
            _op_type = InceptionD
        elif (cur_block / total_blocks) <= 1:
            _op_type = InceptionE
    return _op_type


def _get_dim_type(op):
    match = re.search('\dd', str(op))
    dim_str = match.group(0)
    return dim_str


# =============================================================


def _filter_policy(type, base_dim, max_dim, block_group, cur_dim, total_blocks, policy,
                   parameter_manager):
    if policy == 'default':
        result = base_dim * (2**(block_group))
    elif policy == 'pyramid':
        if type == 'residual-bottleneck':
            ratio = 4
        else:
            ratio = 1
        pyramid_alpha = parameter_manager.get_param('pyramid_alpha', 200)
        result = math.ceil(cur_dim + (pyramid_alpha / total_blocks) * ratio)
    if max_dim != -1:
        return min(max_dim, result)

def AutoConvNet(op,
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
                non_local_start=1e8,
                **kwargs):

    Runner.__model_settings__[
        f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    parameter_manager = ParameterManager(**kwargs)

    base_dim, max_dim = filters
    block_group = 0

    layers = []

    wide_scale = parameter_manager.get_param(
        'wide_scale', 10) if type == 'wide-residual' else 1
    in_dim = input_dim
    out_dim = wide_scale * base_dim

    print(in_dim, out_dim)
    layers.append(layer.__dict__['Conv' + _get_dim_type(op)](
        in_dim, out_dim, 3, 1,
        1))  # Preact first layer is simply a hardcore transform
    layers.append(layer.__dict__['BatchNorm' + _get_dim_type(op)](out_dim))
    in_dim = out_dim

    for i in range(total_blocks):
        id = i + 1
        
        _op_type = _get_op_type(type, id, total_blocks)
        
        if id % pool_freq == 0:
            do_pool = True
            block_group += 1
        else:
            do_pool = False
            
        out_dim = wide_scale * _filter_policy(type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy, parameter_manager)
        
        print(in_dim, out_dim, do_pool)
        kwargs['non_local'] = id>=non_local_start
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
                unit=unit,
                **kwargs))

        in_dim = out_dim

    if flatten:
        layers.append(layer.Flatten())

    # must verify after all keys get registered
    ParameterManager.verify_kwargs(**kwargs)

    return builder(layers)
