from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from ..ops import op
import copy


def push_stem(_op, unit, layers, in_dim, out_dim, stem_type, stem_layers, preact, info, **kwargs):
    stem_kwargs = {}
    stem_kwargs.update(kwargs)
    stem_kwargs['erased_act'] = True if preact else False
    stem_kwargs['preact'] = False
    stem_kwargs['do_norm'] = False if preact else True,
    stem_kwargs['do_pool'] = False
    
    for i in range(stem_layers):
        id = i + 1
        _type = _parse_type(i, stem_type)
        _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim != out_dim)
        
        info.add_row([id, in_dim, out_dim, stem_kwargs['do_pool'], _op_type, 'N/A', 'N/A', 'Stem'])
        
        layers.append(
            _op_type(
                f'{id}',
                unit,
                _op,
                in_dim = in_dim,
                out_dim = out_dim,
                id = id,
                total_blocks = stem_layers,
                **stem_kwargs
            )
        )
        layers.append(op.ManifoldMixup())
        in_dim = out_dim
    return layers, in_dim, out_dim, info
