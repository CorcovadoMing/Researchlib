from ..layers import layer
from .builder import builder
from ..utils import ParameterManager
from ..blocks.unit import unit


def Heads(out_dim, attention=False, **kwargs):
    parameter_manager = ParameterManager(**kwargs)

    last_dim = parameter_manager.get_param('last_dim', None)
    last_dim = last_dim if last_dim else parameter_manager.get_buffer(
        'last_dim')
    dim_type = parameter_manager.get_buffer('dim_type')

    layers = [
        layer.__dict__['BatchNorm' + str(dim_type)](last_dim),
        layer.ReLU(),
        layer.__dict__['DotNonLocalBlock' +
                       str(dim_type)](last_dim) if attention else None,
        unit.conv(layer.__dict__['Conv' + str(dim_type)], last_dim, last_dim,
                  False, True, False) if attention else None,
        layer.__dict__['AdaptiveConcatPool' + str(dim_type)](1),
        layer.Flatten(),
        layer.Linear(2 * last_dim, out_dim)
    ]

    layers = list(filter(None, layers))
    return builder(layers)
