from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from .heads import Heads
from ..runner import Runner
from ..ops import op
from .builder import Builder
from ..utils import ParameterManager
from ..blocks import block
from ..blocks import unit as _unit
from .stem import push_stem
from torch import nn
from texttable import Texttable


def _do_pool(id, pool_freq):
    if (isinstance(pool_freq, int) and id % pool_freq == 0) \
    or (isinstance(pool_freq, list) and id in pool_freq):
        do_pool = True
    else:
        do_pool = False
    return do_pool


def AutoConvNet(
    _op,
    unit,
    input_dim,
    total_blocks,
    type = 'residual',
    filters = (128, 1024),
    filter_policy = 'default',
    stem = {'vgg': 1},
    preact = False,
    pool_freq = 1,
    do_norm = True,
    custom = {},
    **kwargs
):
    Runner.__model_settings__[f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()
    
    info = Texttable(max_width = 0)
    info.add_row(['ID', 'In Dim', 'Out Dim', 'Do Pool?', 'Block Type', 'Keep Output?', 'Keep Input?', 'Groups'])
    
    parameter_manager = ParameterManager(**kwargs)
    share_group_banks = parameter_manager.get_param('share_group_banks', 0)
     
    base_dim, max_dim = filters
    block_group = 0
    layers = []

    # Input mixup
    layers.append(op.ManifoldMixup())

    in_dim = input_dim
    out_dim = base_dim

    # Stem
    if stem is not None:
        stem_type, stem_layers = list(stem.items())[0]
        layers, in_dim, out_dim, info = push_stem(
            _op, _unit.Conv, layers, in_dim, out_dim, stem_type, stem_layers, preact, info, **kwargs
        )
    else:
        stem_layers = 0

    # Body
    cur_bank_dim = -1
    cur_bank = None
    cur_bank_to_manifold = None
    cur_bank_from_manifold = None
    for i in range(total_blocks):
        id = i + 1
        
        do_pool = _do_pool(id, pool_freq)
        if do_pool:
            block_group += 1
            keep_output = False
        else:
            keep_output = _do_pool(id+1, pool_freq) or id == total_blocks
            
        keep_input = True if id == 1 else _do_pool(id-1, pool_freq)

        _type = _parse_type(i, type)
        wide_scale = parameter_manager.get_param('wide_scale', 10) if _type == 'wide-residual' else 1
        
        out_dim = wide_scale * _filter_policy(
            id, type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy,
            parameter_manager
        )
        _op_type = _get_op_type(type, id, total_blocks, do_pool, in_dim != out_dim)
        
        if cur_bank_dim != out_dim and (share_group_banks > 0):
            cur_bank_dim = out_dim
            if _type == 'residual-bottleneck':
                cur_bank_to_manifold = op.TemplateBank(share_group_banks, cur_bank_dim, cur_bank_dim//4, 1)
                cur_bank = op.TemplateBank(share_group_banks, cur_bank_dim//4, cur_bank_dim//4, 3)
                cur_bank_from_manifold = op.TemplateBank(share_group_banks, cur_bank_dim//4, cur_bank_dim, 1)
            else:
                cur_bank = op.TemplateBank(share_group_banks, cur_bank_dim, cur_bank_dim, 3)
                cur_bank_to_manifold = None
                cur_bank_from_manifold = None
        
        # Customization
        layer_kwargs = dict(
            in_dim = in_dim,
            out_dim = out_dim,
            do_pool = do_pool,
            do_norm = do_norm,
            preact = preact,
            id = id,
            total_blocks = total_blocks,
            keep_input = keep_input,
            keep_output = keep_output,
            bank = cur_bank,
            bank_to_manifold = cur_bank_to_manifold,
            bank_from_manifold = cur_bank_from_manifold,
        )
        layer_kwargs.update(kwargs)
        if 'group' in custom and block_group in custom['group']:
            custom_kwargs = custom['group'][block_group]
            layer_kwargs.update(custom_kwargs)
            custom_kwargs_str = [f'{i}: {j}' for i, j in custom_kwargs.items()]
            custom_kwargs_str = ', '.join(custom_kwargs_str)
        else:
            custom_kwargs = None
            custom_kwargs_str = None
        
        info.add_row([id + stem_layers, in_dim, out_dim, do_pool, _op_type.__name__, custom_kwargs_str, None, block_group])
        
        layers.append(
            _op_type(
                f'{id}',
                unit,
                _op,
                **layer_kwargs
            )
        )
        layers.append(op.ManifoldMixup())
        in_dim = out_dim

    # must verify after all keys get registered
    use_subgraph = parameter_manager.get_param('use_subgraph', False)
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(_op))
    parameter_manager.save_buffer('last_dim', out_dim)
    
    print(info.draw())
    
    if use_subgraph:
        return Builder.Seq(layers)
    else:
        return nn.Sequential(*layers)
