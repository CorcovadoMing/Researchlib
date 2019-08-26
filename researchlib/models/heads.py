from ..layers import layer
from .builder import builder
from ..utils import ParameterManager
from ..blocks.unit import unit

def Heads(out_dim, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    last_dim = parameter_manager.get_buffer('last_dim')
    dim_type = parameter_manager.get_buffer('dim_type')
    
    layers = [
        layer.__dict__['BatchNorm'+str(dim_type)](last_dim),
        layer.ReLU(),
        layer.__dict__['DotNonLocalBlock'+str(dim_type)](last_dim),
        unit.conv(layer.__dict__['Conv'+str(dim_type)], last_dim , last_dim , False, True, False),
        layer.__dict__['AdaptiveMaxPool'+str(dim_type)](1),
        layer.Flatten(),
        layer.Linear(last_dim , out_dim)
    ]
    
    return builder(layers)