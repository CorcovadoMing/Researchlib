from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from .heads import Heads
from ..runner import Runner
from ..layers import layer
from ..wrapper import wrapper
from .builder import Builder
from ..utils import ParameterManager
from ..blocks import block
import copy


def AutoConvNet(
    op,
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
    **kwargs
):

    Runner.__model_settings__[f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    parameter_manager = ParameterManager(**kwargs)

    base_dim, max_dim = filters
    block_group = 0

    layers = []
    # Input mixup
    layers.append(layer.ManifoldMixup())

    auxiliary_classifier = parameter_manager.get_param('auxiliary_classifier', None)

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
            wide_scale = parameter_manager.get_param('wide_scale', 10) if _type == 'wide-residual' else 1
            out_dim *= wide_scale
            _op_type = _get_op_type(stem_type, id, stem_layers, False, in_dim == out_dim)
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
    else:
        stem_layers = 0

    # Body
    for i in range(total_blocks):
        id = i + 1
        if (isinstance(pool_freq, int)
            and id % pool_freq == 0) or (isinstance(pool_freq, list) and id in pool_freq):
            block_group += 1
            do_pool = True
        else:
            do_pool = False

        _type = _parse_type(i, type)
        wide_scale = parameter_manager.get_param(
            'wide_scale', 10
        ) if _type == 'wide-residual' else 1
        out_dim = wide_scale * _filter_policy(
            id, type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy,
            parameter_manager
        )
        _op_type = _get_op_type(type, id, total_blocks, do_pool, in_dim == out_dim)
        print(id + stem_layers, in_dim, out_dim, do_pool)

        if do_pool and auxiliary_classifier is not None:
            parameter_manager.save_buffer('dim_type', _get_dim_type(op))
            parameter_manager.save_buffer('last_dim', in_dim)
            layers.append(
                wrapper.Auxiliary(
                    Builder([
                        Heads(auxiliary_classifier),
                        layer.LogSoftmax(
                            -1
                        )  # TODO (Ming): if not classification? if using softmax not logsoftmax?
                    ])
                )
            )
        kwargs['non_local'] = id >= non_local_start
        layers.append(
            _op_type(
                op,
                in_dim,
                out_dim,
                do_pool = do_pool,
                do_norm = do_norm,
                preact = preact,
                id = id,
                total_blocks = total_blocks,
                unit = unit,
                **kwargs
            )
        )
        layers.append(layer.ManifoldMixup())

        in_dim = out_dim

    # must verify after all keys get registered
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(op))
    parameter_manager.save_buffer('last_dim', out_dim)
    return Builder(layers)
