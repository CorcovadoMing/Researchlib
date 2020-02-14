from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def PyramidNet110(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 54,
        type='residual', 
        filters=(16, -1), 
        pool_freq = [18, 36],
        preact=True,
        erased_act = True,
        preact_final_norm = True,
        filter_policy = 'pyramid',
        pyramid_alpha = 200,
        shortcut = 'padding'
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


def PyramidNet272(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 90,
        type='residual-bottleneck', 
        filters=(16, -1), 
        pool_freq = [30, 60],
        preact=True,
        erased_act = True,
        preact_final_norm = True,
        filter_policy = 'pyramid',
        pyramid_alpha = 200,
        shortcut = 'padding'
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)

