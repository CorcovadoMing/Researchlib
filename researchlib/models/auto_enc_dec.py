from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from .heads import Heads
from ..runner import Runner
from ..layers import layer
from ..wrapper import wrapper
from .builder import Builder
from ..utils import ParameterManager
from ..blocks import block
from torch import nn
import torch
import copy


class _RecurrentBlock(nn.Module):
    def __init__(
        self, begin_block, inner_block, end_block, skip_connection = False, skip_type = 'add'
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.skip_type = skip_type
        self.begin = begin_block
        self.inner = inner_block
        self.end = end_block
        self.begin_mmixup = layer.ManifoldMixup()
        self.inner_mmixup = layer.ManifoldMixup()
        self.end_mmixup = layer.ManifoldMixup()

    def forward(self, x):
        x = self.begin(x)
        x = self.begin_mmixup(x)
        out = self.inner(x)
        out = self.inner_mmixup(out)
        if self.skip_connection:
            if self.skip_type == 'add':
                out = out + x
            elif self.skip_type == 'concat':
                out = torch.cat([out, x], dim = 1)
        out = self.end(out)
        return self.end_mmixup(out)


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
    skip_connection = False,
    skip_type = 'concat',
    **kwargs
):

    Runner.__model_settings__[f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    if skip_type not in ['add', 'concat']:
        raise ValueError("skip_type can only be 'add' or 'concat'")

    parameter_manager = ParameterManager(**kwargs)

    base_dim, max_dim = filters
    block_group = 0

    layers = []
    layers.append(layer.ManifoldMixup())

    in_dim = input_dim
    out_dim = base_dim

    # Stem
    if stem is not None:
        stem_type, stem_layers = list(stem.items())[0]
        for i in range(stem_layers):
            id = i + 1
            if i == 0:
                stem_kwargs = copy.deepcopy(kwargs)
                stem_kwargs['erased_activator'] = True if preact else False
            _type = _parse_type(i, type)
            wide_scale = parameter_manager.get_param(
                'wide_scale', 10
            ) if _type == 'wide-residual' else 1
            out_dim *= wide_scale
            _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim == out_dim)
            print(id, in_dim, out_dim, stem_type)
            layers.append(
                _op_type(
                    down_op,
                    in_dim,
                    out_dim,
                    do_pool = False,
                    do_norm = do_norm,
                    preact = False,
                    id = id,
                    total_blocks = stem_layers,
                    unit = unit,
                    **stem_kwargs
                )
            )
            layers.append(layer.ManifoldMixup())
            in_dim = out_dim
    else:
        stem_layers = 0

    # The builder logic is from the middle blocks and recursive to append the begin and end block
    # We calculate the half-part of the model shape first
    dim_cache = []
    for i in range(total_blocks):
        id = i + 1
        if (isinstance(pool_freq, int)
            and id % pool_freq == 0) or (isinstance(pool_freq, list) and id in pool_freq):
            block_group += 1
            do_pool = True
        else:
            do_pool = False

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

        # TODO (Ming): Add an option to use different type of blocks in the autoencoder-like architecture
        _type = _parse_type(i, type)
        wide_scale = parameter_manager.get_param(
            'wide_scale', 10
        ) if _type == 'wide-residual' else 1
        out_dim = wide_scale * _filter_policy(
            id, type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy,
            parameter_manager
        )
        _op_type = _get_op_type(type, id, total_blocks, do_pool, in_dim == out_dim)

        _op_type_begin = _op_type
        _op_type_inner = _op_type
        _op_type_end = _op_type
        # End of TODO

        cache_id, in_dim, out_dim, do_pool = dim_cache.pop()
        end_in_dim = 2 * out_dim if skip_type == 'concat' and skip_connection else out_dim
        print(2 * total_blocks + 1 - cache_id + stem_layers, end_in_dim, in_dim, do_pool)
        kwargs['non_local'] = id >= non_local_start
        structure = _RecurrentBlock(
            # Begin
            _op_type_begin(down_op, in_dim, out_dim,
                do_pool=do_pool, do_norm=do_norm, preact=preact,
                id=cache_id, total_blocks=2*total_blocks+1,
                unit=unit, **kwargs),

            # Inner
            _op_type_inner(down_op, out_dim, out_dim,
                do_pool=False, do_norm=do_norm, preact=preact,
                id=cache_id+1, total_blocks=2*total_blocks+1,
                unit=unit, **kwargs) \
            if id == 1 else structure,

            # End
            _op_type_end(up_op, end_in_dim, in_dim,
                do_pool=do_pool, do_norm=do_norm, preact=preact,
                id=2*total_blocks+1-cache_id, total_blocks=2*total_blocks+1,
                unit=unit, **kwargs),

            skip_connection=skip_connection,
            skip_type=skip_type
        )

    layers.append(structure)

    # must verify after all keys get registered
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(down_op))
    parameter_manager.save_buffer('last_dim', in_dim)
    return Builder.Seq(layers)
