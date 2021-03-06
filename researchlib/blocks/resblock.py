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
                          erased_act=True if (config.preact and config.erased_act) or config.preact_bn_shared else False,
                          padding=config.padding,
                          do_pool=False,
                          do_norm=False if config.preact_bn_shared else config.do_norm,
                          do_share_banks=config.do_share_banks)

    second_conv_kwargs = get_conv_config()
    second_conv_kwargs.update(**kwargs)
    second_conv_kwargs.update(do_pool=False,
                              padding=config.padding,
                              erased_act=not config.preact,
                              do_share_banks=config.do_share_banks)


    conv_op = [
        config._unit(f'{config.prefix}_m1', config._op, config.in_dim, config.out_dim, **first_conv_kwargs),
        op.Downsample(channels = config.out_dim, filt_size = 3, stride = config.stride) if config.blur else None,
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
        
    
class _ResBlock(nn.Module):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''
    
    def __init__(self, prefix, _unit, _op, in_dim, out_dim, **kwargs):
        super().__init__()
    
        parameter_manager = ParameterManager(**kwargs)

        # Utils
        config = get_config(prefix, _unit, _op, in_dim, out_dim, parameter_manager)

        # Merge op
        merge_act = get_act_op(config.act_type, config.dim, config.out_dim) if not config.preact and not config.erased_act else None
        self.merge_op = nn.Sequential(*list(filter(None, [merge_act])))

        # Shared BN
        preact_bn_shared = parameter_manager.get_param('preact_bn_shared', False) and config.preact and (config.in_dim != config.out_dim or config.do_pool)
        setattr(config, 'preact_bn_shared', preact_bn_shared)
        if preact_bn_shared:
            self.shared_bn_op = nn.Sequential(
                *filter(None,[
                    get_norm_op(config.norm_type, config.dim, config.in_dim),
                    get_act_op(config.act_type, config.dim, config.in_dim) if not config.erased_act else None
                ])
            )
        else:
            self.shared_bn_op = op.NoOp()

        self.branch_op = _branch_function(config, parameter_manager, **kwargs)
        if config.stochastic_depth:
            k = config.id / config.total_blocks
            pl = (1-k) + 0.5 * k
            self.branch_op = BernoulliSkip(self.branch_op, pl)
        self.shortcut_op = get_shortcut_op(config, parameter_manager, **kwargs)
    
    def forward(self, x):
        x = self.shared_bn_op(x)
        shortcut = self.shortcut_op(x)
        x = self.branch_op(x)
        return self.merge_op(x + shortcut)