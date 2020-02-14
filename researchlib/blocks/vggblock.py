from .utils import get_conv_config, get_config
from ..utils import ParameterManager


def _VGGBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
    '''
    parameter_manager = ParameterManager(**kwargs)
    config = get_config(prefix, _unit, _op, in_dim, out_dim, parameter_manager)
    new_kwargs = get_conv_config()
    new_kwargs.update(**kwargs)
    new_kwargs.update(do_share_banks=config.do_share_banks)
    return config._unit(config.prefix, config._op, config.in_dim, config.out_dim, **new_kwargs)