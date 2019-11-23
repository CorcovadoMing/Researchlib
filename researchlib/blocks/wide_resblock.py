from ..ops import op
from torch import nn
from .unit.utils import get_act_op, get_norm_op, is_transpose, get_dim
from ..utils import ParameterManager
from .utils import get_conv_config, padding_shortcut, projection_shortcut, SE_Attention, CBAM_Attention


def _WideResBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''
    # Act
    parameter_manager = ParameterManager(**kwargs)
    act_type = parameter_manager.get_param('act_type', 'relu')
    act_op = get_act_op(act_type)
    
    # Norm
    transpose = is_transpose(_op)
    dim = get_dim(_op)
    do_norm = parameter_manager.get_param('do_norm', True)
    norm_type = parameter_manager.get_param('norm_type', 'batch')
    
    # Blur
    do_pool = parameter_manager.get_param('do_pool', False)
    pool_factor = parameter_manager.get_param('pool_factor', 2)
    blur = parameter_manager.get_param('blur', False) and do_pool
    stride = pool_factor if do_pool else 1
    kernel_size = 2 if transpose and do_pool else 3
    padding = 0 if transpose and do_pool else int((kernel_size - 1) / 2)

    
    first_conv_kwargs = get_conv_config()
    first_conv_kwargs.update(**kwargs)
    first_conv_kwargs.update(kernel_size=kernel_size, 
                      stride=1 if blur else stride, 
                      padding=padding,
                      preact=True,
                      do_pool=False)

    second_conv_kwargs = get_conv_config()
    second_conv_kwargs.update(**kwargs)
    second_conv_kwargs.update(do_pool=False,
                              preact=True,
                              drop_rate=0.3)
                              

    if in_dim != out_dim or do_pool:
        first_conv_kwargs.update({'do_norm': False, 'do_act': False})
        conv_op = [
            _unit(f'{prefix}_m1', _op, in_dim, out_dim, **first_conv_kwargs),
            op.Downsample(channels = out_dim, filt_size = 3, stride = stride) if blur else None,
            _unit(f'{prefix}_m2', _op, out_dim, out_dim, **second_conv_kwargs), 
        ]
        pre_shared_norm = nn.Sequential(
            get_norm_op(norm_type, dim, in_dim),
            act_op
        )
    else:
        conv_op = [
            _unit(f'{prefix}_m1', _op, in_dim, out_dim, **first_conv_kwargs),
            op.Downsample(channels = out_dim, filt_size = 3, stride = stride) if blur else None,
            _unit(f'{prefix}_m2', _op, out_dim, out_dim, **second_conv_kwargs), 
        ]
        pre_shared_norm = op.NoOp()

    conv_op = nn.Sequential(*list(filter(None, conv_op)))

    # Shortcut
    shortcut_type = parameter_manager.get_param('shortcut', 'projection')
    shortcut_norm = parameter_manager.get_param('shortcut_norm', False)
    if shortcut_type == 'padding':
        shortcut = padding_shortcut(_op, in_dim, out_dim, get_norm_op(norm_type, dim, out_dim), shortcut_norm, do_pool, pool_factor, blur, transpose, stride)
    else:
        shortcut = projection_shortcut(_op, in_dim, out_dim, get_norm_op(norm_type, dim, out_dim), shortcut_norm, do_pool, pool_factor, blur, transpose, stride)

    # Branch attention
    branch_attention = parameter_manager.get_param('branch_attention')
    if branch_attention == 'se':
        attention_op = SE_Attention(out_dim, dim)
    elif branch_attention == 'cbam':
        attention_op = CBAM_Attention(out_dim, dim)
    else:
        attention_op = op.NoOp()
         
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
        shakedrop_op = op.NoOp()


    flow = {
        f'{prefix}_pre': (pre_shared_norm, [f'{prefix}_input']),
        f'{prefix}_conv': conv_op,
        f'{prefix}_attention': attention_op,
        f'{prefix}_shakedrop': shakedrop_op,
        f'{prefix}_shortcut': (shortcut, [f'{prefix}_pre']),
        f'{prefix}_output': (op.Add, [f'{prefix}_shortcut', f'{prefix}_shakedrop']),
    }

    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=f'{prefix}_output')
