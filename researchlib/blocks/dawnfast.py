from .utils import get_conv_config, SE_Attention, CBAM_Attention
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
        attention_op = nn.Sequential()
         
    # Shakedrop
    shakedrop = parameter_manager.get_param('shakedrop', False)
    if shakedrop:
        id = parameter_manager.get_param('id', required = True)
        total_blocks = parameter_manager.get_param('total_blocks', required = True)
        alpha_range = parameter_manager.get_param('alpha_range', init_value = [-1, 1])
        beta_range = parameter_manager.get_param('beta_range', init_value = [0, 1])
        shakedrop_prob = parameter_manager.get_param('shakedrop_prob', init_value = 0.5)
        mode_mapping = {'batch': 0, 'sample': 1, 'channel': 2, 'pixel': 3}
        mode = parameter_manager.get_param('shakedrop_mode', 'pixel')
        mode = mode_mapping[mode]
        shakedrop_op = op.ShakeDrop(
            id,
            total_blocks,
            alpha_range = alpha_range,
            beta_range = beta_range,
            shakedrop_prob = shakedrop_prob,
            mode = mode
        )
    else:
        shakedrop_op = nn.Sequential()
    
    flow = {
        f'{prefix}_op1': (op1, [f'{prefix}_input']),
        f'{prefix}_op2': op2,
        f'{prefix}_op3': op3,
        f'{prefix}_attention': attention_op,
        f'{prefix}_shakedrop': shakedrop_op,
        f'{prefix}_output': (op.Add, [f'{prefix}_op1', f'{prefix}_shakedrop'])
    }

    return op.Subgraph(flow, f'{prefix}_input', f'{prefix}_output')
