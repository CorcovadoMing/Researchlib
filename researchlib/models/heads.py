from ..layers import layer
from .builder import Builder
from ..utils import ParameterManager
from ..blocks.unit import unit


def Heads(out_dim, attention = False, preact = False, reduce_type = 'concat', **kwargs):
    parameter_manager = ParameterManager(**kwargs)

    last_dim = parameter_manager.get_param('last_dim', None)
    last_dim = last_dim if last_dim else parameter_manager.get_buffer('last_dim')
    dim_type = parameter_manager.get_buffer('dim_type')

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
        layer.__dict__[f'BatchNorm{dim_type}'](last_dim) if preact else None,
        layer.ReLU() if preact else None,

        # Attention
        layer.__dict__[f'DotNonLocalBlock{dim_type}'](last_dim) if attention else None,
        unit.conv(layer.__dict__['Conv' + str(dim_type)], last_dim, last_dim, False, True, False)
        if attention else None,

        # Normal heads
        layer.__dict__[f'Adaptive{_reduce_name}Pool{dim_type}'](1),
        layer.Flatten(),
        layer.Linear(_reduce_dim_calibrate * last_dim, out_dim, bias = False)
    ]

    layers = list(filter(None, layers))
    return Builder(layers)
