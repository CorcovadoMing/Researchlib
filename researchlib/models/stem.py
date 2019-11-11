from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from ..ops import op
import copy


def push_stem(_op, unit, layers, in_dim, out_dim, stem_type, stem_layers, preact, **kwargs):
    stem_kwargs = copy.deepcopy(kwargs)
    stem_kwargs['erased_activator'] = True if preact else False
    if 'stem_pool' not in stem_kwargs:
        stem_kwargs['stem_pool'] = False
    if 'wide_scale' not in stem_kwargs:
        stem_kwargs['wide_scale'] = 10
        
    for i in range(stem_layers):
        id = i + 1
        _type = _parse_type(i, stem_type)
        wide_scale = stem_kwargs['wide_scale'] if _type == 'wide-residual' else 1
        _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim == out_dim)
        out_dim *= wide_scale
        print(id, in_dim, out_dim, stem_type)
        layers.append(
            _op_type(
                _op,
                in_dim,
                out_dim,
                do_pool = stem_kwargs['stem_pool'],
                do_norm = False if preact else True,
                preact = False,
                id = id,
                total_blocks = stem_layers,
                unit = unit,
                **stem_kwargs
            )
        )
        layers.append(op.ManifoldMixup())
        in_dim = out_dim
    return layers, in_dim, out_dim
