from ...ops import op
import torch.nn.utils.spectral_norm as sn
from ...utils import ParameterManager
from .utils import is_transpose, get_dim, get_norm_op, get_act_op, get_pool_op, get_conv_hparams


def _Conv(prefix, _op, in_dim, out_dim, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    do_pool = parameter_manager.get_param('do_pool', False)
    do_norm = parameter_manager.get_param('do_norm', True)
    do_act = parameter_manager.get_param('do_act', True)
    
    transpose = is_transpose(_op)
    dim = get_dim(_op)
    
    preact = parameter_manager.get_param('preact', False)
    prepool = parameter_manager.get_param('prepool', False)
    
    pool_type = parameter_manager.get_param('pool_type', 'upsample') if transpose \
           else parameter_manager.get_param('pool_type', 'max')
    pool_factor = parameter_manager.get_param('pool_factor', 2)
        
    act_type = parameter_manager.get_param('act_type', 'relu')
    norm_type = parameter_manager.get_param('norm_type', 'batch')
    freeze_scale = parameter_manager.get_param('freeze_scale', False)
    freeze_bias = parameter_manager.get_param('freeze_bias', False)
    erased_act = parameter_manager.get_param('erased_act', False)
    
    drop_rate = parameter_manager.get_param('drop_rate', 0)
    dropout_op = op.Dropout(drop_rate, inplace=True) if drop_rate > 0 else None
    
    spectral_norm = parameter_manager.get_param('sn', False)

    
    # Build
    conv_op = _op(in_dim, out_dim, **get_conv_hparams(**kwargs))
    conv_op = sn(conv_op) if spectral_norm else conv_op
    norm_op = None if not do_norm else get_norm_op(norm_type, dim, in_dim if preact else out_dim)
    if freeze_scale:
        norm_op.weight.requires_grad = False
    if freeze_bias:
        norm_op.bias.requires_grad = False
    act_op = None if erased_act or not do_act else get_act_op(act_type)
    pool_op = None if not do_pool else get_pool_op(pool_type, dim, pool_factor)
    
    if preact:
        ops = [norm_op, act_op, dropout_op, conv_op, pool_op]
        names = ['norm', 'act', 'dropout', 'conv', 'pool']
    else:
        if prepool:
            ops = [conv_op, pool_op, norm_op, act_op, dropout_op]
            names = ['conv', 'pool', 'norm', 'act', 'dropout']
        else:
            ops = [conv_op, norm_op, act_op, dropout_op, pool_op]
            names = ['conv', 'norm', 'act', 'dropout', 'pool']
            
    ops = filter(lambda x: x[0] is not None, zip(ops, names))
    
    flow = {f'{prefix}_{k}': v for v, k in ops}
    first, last = list(flow.keys())[0], list(flow.keys())[-1]
    flow[first] = (flow[first], [f'{prefix}_input'])

    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=last)