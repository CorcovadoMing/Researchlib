from ...ops import op
from ...utils import ParameterManager
from functools import partial
import re


def is_transpose(_op):
    match = re.search('Transpose', str(_op))
    return match is not None


def get_dim(_op):
    match = re.search('\dd', str(_op))
    return match.group(0)


def _return_op(mapping, key):
    if type(key) == str:
        if key not in mapping:
            raise ValueError(f'Unknown type {key}')
        else:
            return mapping[key]
    else:
        return key

    
def get_norm_op(norm_type, dim, channel):
    mapping = {
        'batch': op.__dict__[f'BatchNorm{dim}'],
        'instance': op.__dict__[f'InstanceNorm{dim}']
    }
    return _return_op(mapping, norm_type)(channel)


def get_act_op(act_type):
    mapping = {
        'relu': partial(op.ReLU, inplace=True),
        'celu': partial(op.CELU, alpha=0.3, inplace=True)
    }
    return _return_op(mapping, act_type)()


def get_pool_op(pool_type, dim, pool_factor):
    mapping = {
        'max': op.__dict__[f'MaxPool{dim}'],
        'avg': op.__dict__[f'AvgPool{dim}']
    }
    return _return_op(mapping, pool_type)(pool_factor)


def get_conv_hparams(**kwargs):
    parameter_manager = ParameterManager(**kwargs)
    kernel_size = parameter_manager.get_param('kernel_size', required=True)
    stride = parameter_manager.get_param('stride', required=True)
    padding = parameter_manager.get_param('padding', required=True)
    dilation = parameter_manager.get_param('dilation', required=True)
    groups = parameter_manager.get_param('groups', required=True)
    bias = parameter_manager.get_param('bias', required=True)
    return {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups,
        'bias': bias
    }