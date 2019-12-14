from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from .heads import Heads
from ..runner import Runner
from ..ops import op
from ..wrapper import wrapper
from .builder import Builder
from ..utils import ParameterManager
from ..blocks import block
from ..blocks import unit as _unit
from torch import nn
import torch
from .stem import push_stem


def _check_dual_path(var):
    if type(var) == tuple or type(var) == list:
        var, bottleneck = var
    else:
        bottleneck = None
    return var, bottleneck

        
class _RecurrentBlock(nn.Module):
    def __init__(self, begin_block, inner_block, end_block, skip_type = None, return_bottleneck = False):
        super().__init__()
        self.skip_type = skip_type
        self.return_bottleneck = return_bottleneck
        self.return_both = True
        self.is_final = False
        self.begin = begin_block
        self.inner = inner_block
        self.end = end_block
        self.begin_mmixup = op.ManifoldMixup()
        self.inner_mmixup = op.ManifoldMixup()
        self.end_mmixup = op.ManifoldMixup()

    def forward(self, x):
        x, bottleneck = _check_dual_path(x)
        
        x = self.begin(x)
        x = self.begin_mmixup(x)
        out = self.inner(x)

        out, bottleneck = _check_dual_path(out)
            
        bottleneck = out if self.return_bottleneck else bottleneck
        
        out = self.inner_mmixup(out)
        if self.skip_type == 'add':
            out = out + x
        elif self.skip_type == 'concat':
            out = torch.cat([out, x], dim = 1)
        out = self.end(out)
        
        if self.return_both:
            return self.end_mmixup(out), bottleneck
        else:
            return self.end_mmixup(out)


def _do_pool(id, pool_freq):
    if (isinstance(pool_freq, int) and id % pool_freq == 0) \
    or (isinstance(pool_freq, list) and id in pool_freq):
        do_pool = True
    else:
        do_pool = False
    return do_pool


def AutoEncDec(
    down_op,
    up_op,
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
    non_local_start = 1e8,
    skip_type = None,
    return_bottleneck = False,
    **kwargs
):
    Runner.__model_settings__[f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()
    
    parameter_manager = ParameterManager(**kwargs)
    
    base_dim, max_dim = filters
    block_group = 0
    layers = []

    # Input mixup
    layers.append(op.ManifoldMixup())

    in_dim = input_dim
    out_dim = base_dim

    if skip_type not in [None, 'add', 'concat']:
        raise ValueError('skip_type can only be one of None/add/concat')

    # Stem
    if stem is not None:
        stem_type, stem_layers = list(stem.items())[0]
        layers, in_dim, out_dim = push_stem(
            down_op, _unit.Conv, layers, in_dim, out_dim, stem_type, stem_layers, preact, **kwargs
        )
    else:
        stem_layers = 0

    # The builder logic is from the middle blocks and recursive to append the begin and end block
    # We calculate the half-part of the model shape first
    dim_cache = []
    for i in range(total_blocks):
        id = i + 1
        do_pool = _do_pool(id, pool_freq)
        if do_pool:
            block_group += 1

        _type = _parse_type(i, type)
        wide_scale = parameter_manager.get_param('wide_scale', 10) if _type == 'wide-residual' else 1
        
        out_dim = wide_scale * _filter_policy(
            id, type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy,
            parameter_manager
        )
        
        print(id + stem_layers, in_dim, out_dim, do_pool)
        dim_cache.append((id + stem_layers, in_dim, out_dim, do_pool))
        in_dim = out_dim

    # Start build the model recursively
    for i in range(total_blocks):
        id = i + 1
        cache_id, in_dim, out_dim, do_pool = dim_cache.pop()

        # TODO (Ming): Add an option to use different type of blocks in the autoencoder-like architecture
        # Modification targets: _op_type for begin, inner and end
        _type = _parse_type(i, type)
        
        _op_type = _get_op_type(type, id, total_blocks, do_pool, in_dim == out_dim)
        _op_type_begin = _op_type
        _op_type_inner = _op_type
        _op_type_end = _op_type
        # End of TODO

        end_in_dim = 2 * out_dim if skip_type == 'concat' and do_pool else out_dim
        _skip_type = skip_type if do_pool else None
        if id == 1:
            print('Bottleneck', out_dim, out_dim)
        print(2 * total_blocks + 2 - cache_id + stem_layers, end_in_dim, in_dim, do_pool, _skip_type)
        kwargs['non_local'] = id >= non_local_start
        structure = _RecurrentBlock(
            # Begin
            _op_type_begin(f'{cache_id}', unit, down_op, in_dim, out_dim,
                do_pool=do_pool, do_norm=do_norm, preact=preact,
                id=cache_id, total_blocks=2*total_blocks+1, **kwargs),

            # Inner
            _op_type_inner(f'{cache_id}', unit, down_op, out_dim, out_dim,
                do_pool=False, do_norm=do_norm, preact=preact,
                id=cache_id+1, total_blocks=2*total_blocks+1, **kwargs) \
            if id == 1 else structure,

            # End
            _op_type_end(f'{cache_id}', unit, up_op, end_in_dim, in_dim,
                do_pool=do_pool, do_norm=do_norm, preact=preact,
                id=2*total_blocks+1-cache_id, total_blocks=2*total_blocks+1, **kwargs),

            skip_type=_skip_type,
            return_bottleneck=return_bottleneck and id==1
        )

    structure.return_both = return_bottleneck
    layers.append(structure)

    # must verify after all keys get registered
    parameter_manager.allow_param('non_local')
    use_subgraph = parameter_manager.get_param('use_subgraph', False)
    
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(down_op))
    parameter_manager.save_buffer('last_dim', in_dim)
    
    if use_subgraph:
        return Builder.Seq(layers)
    else:
        return nn.Sequential(*layers)
