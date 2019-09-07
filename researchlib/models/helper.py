import re
import math
from ..blocks import block


def _get_op_type(type, cur_block, total_blocks, do_pool, do_expand):
    if type not in [
            'vgg', 'residual', 'residual-bottleneck', 'wide-residual',
            'inverted-bottleneck', 'inceptionv3', 'inceptionv4',
            'inception-residualv2'
    ]:
        raise ('Type is not supperted')
    if type == 'vgg':
        _op_type = block.VGGBlock
    elif type == 'residual':
        _op_type = block.ResBlock
    elif type == 'residual-bottleneck':
        _op_type = block.ResBottleneckBlock
    elif type == 'wide-residual':
        _op_type = block.WideResBlock
    elif type == 'inverted-bottleneck':
        _op_type = block.InvertedBottleneckBlock
    elif type == 'inceptionv3':
        if (cur_block / total_blocks) <= 0.2:
            _op_type = block.InceptionV3A
        elif (cur_block / total_blocks) <= 0.4:
            _op_type = block.InceptionV3B
        elif (cur_block / total_blocks) <= 0.6:
            _op_type = block.InceptionV3C
        elif (cur_block / total_blocks) <= 0.8:
            _op_type = block.InceptionV3D
        elif (cur_block / total_blocks) <= 1:
            _op_type = block.InceptionV3E
    elif type == 'inceptionv4':
        if (cur_block / total_blocks) <= (1 / 3):
            _op_type = block.ReductionV4A if do_pool else block.InceptionV4A
        elif (cur_block / total_blocks) <= (2 / 3):
            _op_type = block.ReductionV4B if do_pool else block.InceptionV4B
        elif (cur_block / total_blocks) <= 1:
            # We don't have Reduction type C
            _op_type = block.ReductionV4B if do_pool else block.InceptionV4C
    elif type == 'inception-residualv2':
        if (cur_block / total_blocks) <= (1 / 3):
            _op_type = block.ReductionV2A if do_pool or do_expand else block.InceptionResidualV2A
        elif (cur_block / total_blocks) <= (2 / 3):
            _op_type = block.ReductionV2B if do_pool or do_expand else block.InceptionResidualV2B
        elif (cur_block / total_blocks) <= 1:
            # We don't have Reduction type C
            _op_type = block.ReductionV2B if do_pool or do_expand else block.InceptionResidualV2C
    return _op_type


def _get_dim_type(op):
    match = re.search('\dd', str(op))
    dim_str = match.group(0)
    return dim_str


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