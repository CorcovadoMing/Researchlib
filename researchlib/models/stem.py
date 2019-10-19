from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from ..utils import ParameterManager
from ..layers import layer
import copy


def push_stem(layers, in_dim, out_dim, wide_scale, stem_type, stem_layers, **kwargs)    
    for i in range(stem_layers):
        id = i + 1
        if i == 0:
            stem_kwargs = copy.deepcopy(kwargs)
            stem_kwargs['erased_activator'] = True if preact else False
        _type = _parse_type(i, type)
        _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim == out_dim)
        
        out_dim *= wide_scale
        print(id, in_dim, out_dim, stem_type)
        layers.append(
            _op_type(
                op,
                in_dim,
                out_dim,
                do_pool = False,
                do_norm = False if preact else True,
                preact = False,
                id = id,
                total_blocks = stem_layers,
                unit = unit,
                **stem_kwargs
            )
        )
        layers.append(layer.ManifoldMixup())
        in_dim = out_dim
    return layers, in_dim, out_dim