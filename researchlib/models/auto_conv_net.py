from .helper import _get_dim_type, _filter_policy, _get_op_type
from .heads import Heads
from ..runner import Runner
from ..layers import layer
from ..wrapper import wrapper
from .builder import builder
from ..utils import ParameterManager
from ..blocks import block


def AutoConvNet(op,
                unit,
                input_dim,
                total_blocks,
                type='residual',
                filters=(128, 1024),
                filter_policy='default',
                preact=True,
                pool_freq=1,
                do_norm=True,
                non_local_start=1e8,
                **kwargs):

    Runner.__model_settings__[
        f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()

    parameter_manager = ParameterManager(**kwargs)

    base_dim, max_dim = filters
    block_group = 0

    layers = []

    wide_scale = parameter_manager.get_param('wide_scale', 10) if type == 'wide-residual' else 1
    no_end_pool = parameter_manager.get_param('no_end_pool', False)
    auxiliary_classifier = parameter_manager.get_param('auxiliary_classifier', None)
    
    in_dim = input_dim
    out_dim = wide_scale * base_dim  

    print(in_dim, out_dim)
    layers.append(layer.__dict__['Conv' + _get_dim_type(op)](in_dim, out_dim, 3, 1, 1))  # Preact first layer is simply a hardcore transform
    layers.append(layer.__dict__['BatchNorm' + _get_dim_type(op)](out_dim))
    in_dim = out_dim

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

        _op_type = _get_op_type(type, id, total_blocks, do_pool,
                                in_dim == out_dim)

        print(in_dim, out_dim, do_pool)
        if do_pool and auxiliary_classifier is not None:
            parameter_manager.save_buffer('dim_type', _get_dim_type(op))
            parameter_manager.save_buffer('last_dim', in_dim)
            layers.append(
                wrapper.Auxiliary(
                    builder([
                        Heads(auxiliary_classifier),
                        layer.LogSoftmax(-1) # TODO (Ming): if not classification? if using softmax not logsoftmax?
                    ])
                ))
        kwargs['non_local'] = id >= non_local_start
        layers.append(
            _op_type(
                op,
                in_dim,
                out_dim,
                do_pool=do_pool,
                do_norm=do_norm,
                preact=preact,
                id=id,
                total_blocks=total_blocks,
                unit=unit,
                **kwargs))

        in_dim = out_dim

    # must verify after all keys get registered
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(op))
    parameter_manager.save_buffer('last_dim', out_dim)
    return builder(layers)
