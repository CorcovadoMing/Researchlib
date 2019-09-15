from .helper import _get_dim_type, _filter_policy, _get_op_type
from .heads import Heads
from ..runner import Runner
from ..layers import layer
from ..wrapper import wrapper
from .builder import builder
from ..utils import ParameterManager
from ..blocks import block
from torch import nn
import torch
import copy


class _RecurrentBlock(nn.Module):

    def __init__(self,
                 begin_block,
                 inner_block,
                 end_block,
                 skip_connection=False,
                 skip_type='add'):
        super().__init__()
        self.skip_connection = skip_connection
        self.skip_type = skip_type
        self.begin = begin_block
        self.inner = inner_block
        self.end = end_block

    def forward(self, x):
        x = self.begin(x)
        out = self.inner(x)
        if self.skip_connection:
            if self.skip_type == 'add':
                out = out + x
            elif self.skip_type == 'concat':
                out = torch.cat([out, x], dim=1)
        return self.end(out)


def AutoEncDec(down_op,
               up_op,
               unit,
               input_dim,
               total_blocks,
               type='residual',
               filters=(128, 1024),
               filter_policy='default',
               stem={'vgg': 1},
               preact=True,
               pool_freq=1,
               do_norm=True,
               non_local_start=1e8,
               skip_connection=False,
               skip_type='concat',
               **kwargs):

    Runner.__model_settings__[
        f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    if skip_type not in ['add', 'concat']:
        raise ValueError("skip_type can only be 'add' or 'concat'")

    parameter_manager = ParameterManager(**kwargs)

    base_dim, max_dim = filters
    block_group = 0

    layers = []

    wide_scale = parameter_manager.get_param(
        'wide_scale', 10) if type == 'wide-residual' else 1
    no_end_pool = parameter_manager.get_param('no_end_pool', False)

    in_dim = input_dim
    out_dim = wide_scale * base_dim

    # Stem
    stem_type, stem_layers = list(stem.items())[0]
    for i in range(stem_layers):
        id = i + 1
        print(id, in_dim, out_dim, stem_type)
        if i == 0:
            stem_kwargs = copy.deepcopy(kwargs)
            stem_kwargs['erased_activator'] = True if preact else False
        _op_type = _get_op_type(stem_type, i, stem_layers, False,
                                in_dim == out_dim)
        layers.append(
            _op_type(
                down_op,
                in_dim,
                out_dim,
                do_pool=False,
                do_norm=do_norm,
                preact=False,
                id=id,
                total_blocks=stem_layers,
                unit=unit,
                **stem_kwargs))
        in_dim = out_dim

    # The builder logic is from the middle blocks and recursive to append the begin and end block
    # We calculate the half-part of the model shape first
    dim_cache = []
    for i in range(total_blocks):
        id = i + 1
        if id % pool_freq == 0:
            block_group += 1
            do_pool = False if id == total_blocks and no_end_pool else True
        else:
            do_pool = False

        out_dim = wide_scale * _filter_policy(id, type, base_dim, max_dim,
                                              block_group, in_dim, total_blocks,
                                              filter_policy, parameter_manager)
        print(id + stem_layers, in_dim, out_dim, do_pool)
        dim_cache.append((id + stem_layers, in_dim, out_dim, do_pool))
        in_dim = out_dim

    # Start build the model recursively
    for i in range(total_blocks):
        # TODO (Ming): id needs to be fixed, it doesn't work in this kinds of architecture
        id = i + 1

        _op_type_begin = _get_op_type(type, id, total_blocks, do_pool,
                                      in_dim == out_dim)
        _op_type_inner = _get_op_type(type, id, total_blocks, do_pool,
                                      in_dim == out_dim)
        _op_type_end = _get_op_type(type, id, total_blocks, do_pool,
                                    in_dim == out_dim)

        cache_id, in_dim, out_dim, do_pool = dim_cache.pop()
        end_in_dim = out_dim if skip_type == 'add' else 2 * out_dim
        print(2 * total_blocks + 1 - cache_id + stem_layers, end_in_dim, in_dim,
              do_pool)
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
    return builder(layers)
