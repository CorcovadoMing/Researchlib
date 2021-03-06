from .utils import get_conv_config, SE_Attention, CBAM_Attention, get_shakedrop_op
from ..utils import ParameterManager
from ..ops import op
from torch import nn


def _DAWNBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
        https://myrtle.ai/how-to-train-your-resnet-4-architecture/
    '''
    parameter_manager = ParameterManager(**kwargs)
    
    first_conv_kwargs = get_conv_config()
    first_conv_kwargs.update(**kwargs)
    first_conv_kwargs.update(preact=False)
    
    remains_conv_kwargs = get_conv_config()
    remains_conv_kwargs.update(**kwargs)
    remains_conv_kwargs.update(preact=False, do_pool=False)
    
    op1 = _unit(f'{prefix}_m1', _op, in_dim, out_dim, **first_conv_kwargs)
    op2 = _unit(f'{prefix}_m2', _op, out_dim, out_dim, **remains_conv_kwargs)
    op3 = _unit(f'{prefix}_m3', _op, out_dim, out_dim, **remains_conv_kwargs)
    
    # Branch attention
    branch_attention = parameter_manager.get_param('branch_attention')
    if branch_attention == 'se':
        attention_op = SE_Attention(out_dim, dim)
    elif branch_attention == 'cbam':
        attention_op = CBAM_Attention(out_dim, dim)
    else:
        attention_op = op.NoOp()
         
    # Shakedrop
    shakedrop_op = get_shakedrop_op(parameter_manager)
    
    flow = {
        f'{prefix}_op1': (op1, [f'{prefix}_input']),
        f'{prefix}_op2': op2,
        f'{prefix}_op3': op3,
        f'{prefix}_attention': attention_op,
        f'{prefix}_shakedrop': shakedrop_op,
        f'{prefix}_output': (op.Add, [f'{prefix}_op1', f'{prefix}_shakedrop'])
    }

    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=f'{prefix}_output')
