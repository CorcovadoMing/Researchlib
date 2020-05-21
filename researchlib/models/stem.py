from .helper import _get_dim_type, _filter_policy, _get_op_type
from ..ops import op
import copy


def push_large_stem(_op, unit, layers, in_dim, out_dim, stem_type, stem_layers, preact, info, **kwargs):
    final_stem_kwargs = {}
    final_stem_kwargs.update(kwargs)
    final_stem_kwargs['erased_act'] = True if preact else False
    final_stem_kwargs['preact'] = False
    final_stem_kwargs['do_norm'] = False if preact else True
    final_stem_kwargs['do_pool'] = False
    
    normal_stem_kwargs = {}
    normal_stem_kwargs.update(kwargs)
    normal_stem_kwargs['erased_act'] = False
    normal_stem_kwargs['preact'] = False
    normal_stem_kwargs['do_norm'] = True
    normal_stem_kwargs['do_pool'] = False
    
    id = 1
    _in_dim = in_dim
    _out_dim = out_dim // 2
    _op_type = _get_op_type(stem_type, id, stem_layers, False, _in_dim != _out_dim)
    info.add_row([id, _in_dim, _out_dim, True, _op_type.__name__, 'N/A', 'N/A', 'Stem'])
    layers.append(
        _op_type(
            f'{id}',
            unit,
            _op,
            in_dim = _in_dim,
            out_dim = _out_dim,
            id = id,
            total_blocks = stem_layers,
            stride=2,
            **normal_stem_kwargs
        )
    )
    
    id = 2
    _in_dim = _out_dim
    _op_type = _get_op_type(stem_type, id, stem_layers, False, _in_dim != _out_dim)
    info.add_row([id, _in_dim, _out_dim, False, _op_type.__name__, 'N/A', 'N/A', 'Stem'])
    layers.append(
        _op_type(
            f'{id}',
            unit,
            _op,
            in_dim = _in_dim,
            out_dim = _out_dim,
            id = id,
            total_blocks = stem_layers,
            **normal_stem_kwargs
        )
    )
    
    id = 3
    _in_dim = _out_dim
    _out_dim = out_dim
    _op_type = _get_op_type(stem_type, id, stem_layers, False, _in_dim != _out_dim)
    info.add_row([id, _in_dim, _out_dim, True, _op_type.__name__, 'N/A', 'N/A', 'Stem'])
    layers.append(
        _op_type(
            f'{id}',
            unit,
            _op,
            in_dim = _in_dim,
            out_dim = _out_dim,
            id = id,
            total_blocks = stem_layers,
            **final_stem_kwargs
        )
    )
    
    layers.append(op.__dict__[f'MaxPool{_get_dim_type(_op)}'](3, 2, 1))
    in_dim = out_dim
    
    return layers, in_dim, out_dim, info


def push_small_stem(_op, unit, layers, in_dim, out_dim, stem_type, stem_layers, preact, info, **kwargs):
    stem_kwargs = {}
    stem_kwargs.update(kwargs)
    stem_kwargs['erased_act'] = True if preact else False
    stem_kwargs['preact'] = False
    stem_kwargs['do_norm'] = False if preact else True
    stem_kwargs['do_pool'] = False
    
    for i in range(stem_layers):
        id = i + 1
        _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim != out_dim)
        info.add_row([id, in_dim, out_dim, stem_kwargs['do_pool'], _op_type.__name__, 'N/A', 'N/A', 'Stem'])
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
        in_dim = out_dim

    return layers, in_dim, out_dim, info
