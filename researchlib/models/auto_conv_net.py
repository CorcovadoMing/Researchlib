from .helper import _get_dim_type, _filter_policy, _get_op_type, _parse_type
from .heads import Heads
from ..runner import Runner
from ..ops import op
from ..wrapper import wrapper
from .builder import Builder
from ..utils import ParameterManager
from ..blocks import block
from ..blocks import unit as _unit
from .stem import push_stem
from torch import nn

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
    non_local_start = 1e8,
    **kwargs
):
    Runner.__model_settings__[f'{type}-blocks{total_blocks}_input{input_dim}'] = locals()
    
    parameter_manager = ParameterManager(**kwargs)
    
    auxiliary_classifier = parameter_manager.get_param('auxiliary_classifier', None)
    
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
        layers, in_dim, out_dim = push_stem(
            _op, _unit.Conv, layers, in_dim, out_dim, stem_type, stem_layers, preact, **kwargs
        )
    else:
        stem_layers = 0

    # Body
    for i in range(total_blocks):
        id = i + 1
        do_pool = _do_pool(id, pool_freq)
        if do_pool:
            block_group += 1
            save_output = False
        else:
            if _do_pool(id+1, pool_freq):
                save_output = True
            else:
                save_output = False
        if id == total_blocks:
            save_output = True
        
        if id == 1:
            normal_pass = True
        else:
            if _do_pool(id-1, pool_freq):
                normal_pass = True
            else:
                normal_pass = False
                
            
            

        _type = _parse_type(i, type)
        wide_scale = parameter_manager.get_param('wide_scale', 10) if _type == 'wide-residual' else 1
        
        out_dim = wide_scale * _filter_policy(
            id, type, base_dim, max_dim, block_group, in_dim, total_blocks, filter_policy,
            parameter_manager
        )
        _op_type = _get_op_type(type, id, total_blocks, do_pool, in_dim != out_dim)
        print(id + stem_layers, in_dim, out_dim, do_pool, _op_type, save_output, normal_pass)

        if do_pool and auxiliary_classifier is not None:
            parameter_manager.save_buffer('dim_type', _get_dim_type(_op))
            parameter_manager.save_buffer('last_dim', in_dim)
            layers.append(
                wrapper.Auxiliary(
                    Builder([
                        Heads(auxiliary_classifier),
                        layer.LogSoftmax(-1)
                        # TODO (Ming): if not classification? if using softmax not logsoftmax?
                    ])
                )
            )

        kwargs['non_local'] = id >= non_local_start
        layers.append(
            _op_type(
                f'{id}',
                unit,
                _op,
                in_dim = in_dim,
                out_dim = out_dim,
                do_pool = do_pool,
                do_norm = do_norm,
                preact = preact,
                id = id,
                total_blocks = total_blocks,
                save_output = save_output,
                normal_pass = normal_pass,
                **kwargs
            )
        )
        layers.append(op.ManifoldMixup())

        in_dim = out_dim

    # must verify after all keys get registered
    parameter_manager.allow_param('non_local')
    use_subgraph = parameter_manager.get_param('use_subgraph', False)
    
    ParameterManager.verify_kwargs(**kwargs)
    parameter_manager.save_buffer('dim_type', _get_dim_type(_op))
    parameter_manager.save_buffer('last_dim', out_dim)
    
    if use_subgraph:
        return Builder.Seq(layers)
    else:
        return nn.Sequential(*layers)
