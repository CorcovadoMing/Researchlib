from .utils import get_conv_config

def _VGGBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
    '''
    new_kwargs = get_conv_config()
    new_kwargs.update(**kwargs)
    return _unit(prefix, _op, in_dim, out_dim, **new_kwargs)