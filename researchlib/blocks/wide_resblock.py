from ..ops import op
from torch import nn
from .unit.utils import get_act_op, get_norm_op, is_transpose, get_dim
from ..utils import ParameterManager
from .utils import get_conv_config, padding_shortcut, projection_shortcut, SE_Attention, CBAM_Attention, get_shakedrop_op, get_shortcut_op, get_config, BernoulliSkip

def _branch_function(config, parameter_manager, **kwargs):
    first_conv_kwargs = get_conv_config()
    first_conv_kwargs.update(**kwargs)
    first_conv_kwargs.update(kernel_size=config.kernel_size, 
                      stride=1 if config.blur else config.stride, 
                      padding=config.padding,
                      preact=True,
                      do_pool=False)

    second_conv_kwargs = get_conv_config()
    second_conv_kwargs.update(**kwargs)
    second_conv_kwargs.update(do_pool=False,
                              preact=True,
                              drop_rate=0.3)
                              

    if config.in_dim != config.out_dim or config.do_pool:
        first_conv_kwargs.update({'do_norm': False, 'do_act': False})
        conv_op = [
            config._unit(f'{config.prefix}_m1', config._op, config.in_dim, config.out_dim, **first_conv_kwargs),
            op.Downsample(channels = config.out_dim, filt_size = 3, stride = config.stride) if config.blur else None,
            config._unit(f'{config.prefix}_m2', config._op, config.out_dim, config.out_dim, **second_conv_kwargs), 
        ]
        pre_shared_norm_op = nn.Sequential(
            get_norm_op(config.norm_type, config.dim, config.in_dim),
            get_act_op(config.act_type)
        )
    else:
        conv_op = [
            config._unit(f'{config.prefix}_m1', config._op, config.in_dim, config.out_dim, **first_conv_kwargs),
            op.Downsample(channels = config.out_dim, filt_size = 3, stride = config.stride) if config.blur else None,
            config._unit(f'{config.prefix}_m2', config._op, config.out_dim, config.out_dim, **second_conv_kwargs), 
        ]
        pre_shared_norm_op = op.NoOp()

    conv_op = nn.Sequential(*list(filter(None, conv_op)))
    
    # Branch attention
    branch_attention = parameter_manager.get_param('branch_attention')
    if branch_attention == 'se':
        attention_op = SE_Attention(config.out_dim, config.dim)
    elif branch_attention == 'cbam':
        attention_op = CBAM_Attention(config.out_dim, config.dim)
    else:
        attention_op = op.NoOp()
         
    # Shakedrop
    shakedrop_op = get_shakedrop_op(parameter_manager)
    
    return nn.Sequential(
        conv_op,
        attention_op,
        shakedrop_op
    ), pre_shared_norm_op

    
def _WideResBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''
    
    parameter_manager = ParameterManager(**kwargs)
    
    # Utils
    config = get_config(prefix, _unit, _op, in_dim, out_dim, parameter_manager)
    
    branch_op, pre_shared_norm_op = _branch_function(config, parameter_manager, **kwargs)
    if config.stochastic_depth > 0:
        branch_op = BernoulliSkip(branch_op, config.stochastic_depth)
    shortcut_op = get_shortcut_op(config, parameter_manager, **kwargs)


    flow = {
        f'{prefix}_pre': (pre_shared_norm_op, [f'{prefix}_input']),
        f'{prefix}_branch': branch_op,
        f'{prefix}_shortcut': (shortcut_op, [f'{prefix}_pre']),
        f'{prefix}_output': (op.Add, [f'{prefix}_shortcut', f'{prefix}_branch']),
    }

    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=f'{prefix}_output')
