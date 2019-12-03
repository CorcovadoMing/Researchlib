import torch
from ..ops import op
from torch import nn
from .unit.utils import get_act_op, get_norm_op, is_transpose, get_dim
from ..utils import ParameterManager
from .utils import get_conv_config, padding_shortcut, projection_shortcut, SE_Attention, CBAM_Attention, get_shakedrop_op, get_shortcut_op, get_config, BernoulliSkip


def _branch_function(config, parameter_manager, **kwargs):
    # Preact final norm
    preact_final_norm = parameter_manager.get_param('preact_final_norm', False)
    preact_final_norm_op = get_norm_op(config.norm_type, config.dim, config.out_dim) if config.do_norm and config.preact and preact_final_norm else None

    first_conv_kwargs = get_conv_config()
    first_conv_kwargs.update(**kwargs)
    first_conv_kwargs.update(kernel_size=config.kernel_size, 
                      stride=1 if config.blur else config.stride, 
                      padding=config.padding,
                      erased_act=True if (config.preact and config.erased_act) or config.preact_bn_shared else False,
                      do_pool=False,
                      do_norm=False if config.preact_bn_shared else config.do_norm)

    second_conv_kwargs = get_conv_config()
    second_conv_kwargs.update(**kwargs)
    second_conv_kwargs.update(do_pool=False,
                              erased_act=not config.preact)


    conv_op = [
        config._unit(f'{config.prefix}_m1', config._op, config.in_dim, config.out_dim, **first_conv_kwargs),
        op.Downsample(channels = config.out_dim, filt_size = 3, stride = stride) if config.blur else None,
        config._unit(f'{config.prefix}_m2', config._op, config.out_dim, config.out_dim, **second_conv_kwargs), 
        preact_final_norm_op
    ]

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
    )
        

def _ResBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''
    
    parameter_manager = ParameterManager(**kwargs)
    
    # Utils
    config = get_config(prefix, _unit, _op, in_dim, out_dim, parameter_manager)
    
    # Merge op
    merge_act = get_act_op(config.act_type) if not config.preact and not config.erased_act else None
    merge_op = nn.Sequential(*list(filter(None, [merge_act])))
    
    # Shared BN
    preact_bn_shared = parameter_manager.get_param('preact_bn_shared', False) and config.preact and (config.in_dim != config.out_dim or config.do_pool)
    setattr(config, 'preact_bn_shared', preact_bn_shared)
    if preact_bn_shared:
        shared_bn_op = nn.Sequential(
            *filter(None,[
                get_norm_op(config.norm_type, config.dim, config.in_dim),
                get_act_op(config.act_type) if not config.erased_act else None
            ])
        )
    else:
        shared_bn_op = op.NoOp()
        
    branch_op = _branch_function(config, parameter_manager, **kwargs)
    if config.stochastic_depth > 0:
        branch_op = BernoulliSkip(branch_op, config.stochastic_depth)
    shortcut_op = get_shortcut_op(config, parameter_manager, **kwargs)


    flow = {
        f'{prefix}_shared_bn': (shared_bn_op, [f'{prefix}_input']),
        f'{prefix}_branch': branch_op,
        f'{prefix}_shortcut': (shortcut_op, [f'{prefix}_shared_bn']),
        f'{prefix}_add': (op.Add, [f'{prefix}_shortcut', f'{prefix}_branch']),
        f'{prefix}_output': merge_op
    }

    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=f'{prefix}_output')
