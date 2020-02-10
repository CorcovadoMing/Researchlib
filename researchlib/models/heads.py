from ..ops import op
from .builder import Builder
from ..utils import ParameterManager
from ..blocks.unit import unit
from torch import nn


def Heads(out_dim, attention = False, preact = False, reduce_type = 'concat', linear_bias = False, channels_transform = False, use_subgraph = False, **kwargs):
    parameter_manager = ParameterManager(**kwargs)

    last_dim = parameter_manager.get_param('last_dim', None)
    last_dim = last_dim if last_dim else parameter_manager.get_buffer('last_dim', clear=False)
    dim_type = parameter_manager.get_param('dim_type', None) or parameter_manager.get_buffer('dim_type', clear=False)

    if reduce_type == 'concat':
        _reduce_name = 'Concat'
        _reduce_dim_calibrate = 2
    elif reduce_type == 'avg':
        _reduce_name = 'Avg'
        _reduce_dim_calibrate = 1
    elif reduce_type == 'max':
        _reduce_name = 'Max'
        _reduce_dim_calibrate = 1
    else:
        raise ValueError(f'{reduce_type} is not supported')

    layers = [
        # Preact
        op.__dict__[f'BatchNorm{dim_type}'](last_dim) if preact else None,

        # Attention
        op.__dict__[f'DotNonLocalBlock{dim_type}'](last_dim) if attention else None,
        unit.Conv(op.__dict__['Conv' + str(dim_type)], last_dim, last_dim, False, True, False)
        if attention else None,

        # Normal heads
        op.__dict__[f'Conv{dim_type}'](last_dim, last_dim, 1, bias = linear_bias) if channels_transform else None,
        op.__dict__[f'Adaptive{_reduce_name}Pool{dim_type}'](1),
        op.Flatten(),
        op.Linear(_reduce_dim_calibrate * last_dim, out_dim, bias = linear_bias)
    ]

    layers = list(filter(None, layers))
    if use_subgraph:
        return Builder.Seq(layers)
    else:
        return nn.Sequential(*layers)
