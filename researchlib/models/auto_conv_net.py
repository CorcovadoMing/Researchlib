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
from ..blocks._inception_v3 import InceptionV3A, InceptionV3B, InceptionV3C, InceptionV3D, InceptionV3E
from ..blocks._inception_v4 import InceptionV4A, InceptionV4B, InceptionV4C, ReductionV4A, ReductionV4B
from ..blocks._inception_residual_v2 import InceptionResidualV2A, InceptionResidualV2B, InceptionResidualV2C, ReductionV2A, ReductionV2B

# =============================================================


def _get_op_type(type, cur_block, total_blocks, do_pool, do_expand):
    if type not in [
            'vgg', 'residual', 'residual-bottleneck', 'wide-residual',
            'inverted-bottleneck', 'inceptionv3', 'inceptionv4',
            'inception-residualv2'
    ]:
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
    elif type == 'inceptionv3':
        if (cur_block / total_blocks) <= 0.2:
            _op_type = InceptionV3A
        elif (cur_block / total_blocks) <= 0.4:
            _op_type = InceptionV3B
        elif (cur_block / total_blocks) <= 0.6:
            _op_type = InceptionV3C
        elif (cur_block / total_blocks) <= 0.8:
            _op_type = InceptionV3D
        elif (cur_block / total_blocks) <= 1:
            _op_type = InceptionV3E
    elif type == 'inceptionv4':
        if (cur_block / total_blocks) <= (1 / 3):
            _op_type = ReductionV4A if do_pool else InceptionV4A
        elif (cur_block / total_blocks) <= (2 / 3):
            _op_type = ReductionV4B if do_pool else InceptionV4B
        elif (cur_block / total_blocks) <= 1:
            # We don't have Reduction type C
            _op_type = ReductionV4B if do_pool else InceptionV4C
    elif type == 'inception-residualv2':
        if (cur_block / total_blocks) <= (1 / 3):
            _op_type = ReductionV2A if do_pool or do_expand else InceptionResidualV2A
        elif (cur_block / total_blocks) <= (2 / 3):
            _op_type = ReductionV2B if do_pool or do_expand else InceptionResidualV2B
        elif (cur_block / total_blocks) <= 1:
            # We don't have Reduction type C
            _op_type = ReductionV2B if do_pool or do_expand else InceptionResidualV2C
    return _op_type


def _get_dim_type(op):
    match = re.search('\dd', str(op))
    dim_str = match.group(0)
    return dim_str


# =============================================================


def _filter_policy(block_idx, type, base_dim, max_dim, block_group, cur_dim,
                   total_blocks, policy, parameter_manager):
    if policy == 'default':
        result = base_dim * (2**(block_group))
    elif policy == 'pyramid':
        if type == 'residual-bottleneck':
            ratio = 4
        else:
            ratio = 1
        pyramid_alpha = parameter_manager.get_param('pyramid_alpha', 200)
        result = math.floor(base_dim +
                            pyramid_alpha * block_idx / total_blocks) * ratio
    if max_dim != -1:
        return min(max_dim, result)
    else:
        return result


def AutoConvNet(op,
                unit,
                input_dim,
                total_blocks,
                type='residual',
                filters=(128, 1024),
                filter_policy='default',
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
    no_end_pool = parameter_manager.get_param('no_end_pool', False)

    print(in_dim, out_dim)
    layers.append(layer.__dict__['Conv' + _get_dim_type(op)](
        in_dim, out_dim, 3, 1,
        1))  # Preact first layer is simply a hardcore transform
    layers.append(layer.__dict__['BatchNorm' + _get_dim_type(op)](out_dim))
    in_dim = out_dim

    for i in range(total_blocks):
        id = i + 1

        if id % pool_freq == 0:
            block_group += 1
            if id == total_blocks and no_end_pool:
                do_pool = False
            else:
                do_pool = True
        else:
            do_pool = False

        out_dim = wide_scale * _filter_policy(id, type, base_dim, max_dim,
                                              block_group, in_dim, total_blocks,
                                              filter_policy, parameter_manager)

        _op_type = _get_op_type(type, id, total_blocks, do_pool,
                                in_dim == out_dim)

        print(in_dim, out_dim, do_pool)
        kwargs['non_local'] = id >= non_local_start
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

    # must verify after all keys get registered
    ParameterManager.verify_kwargs(**kwargs)

    parameter_manager.save_buffer('last_dim', out_dim)
    parameter_manager.save_buffer('dim_type', _get_dim_type(op))

    return builder(layers)
