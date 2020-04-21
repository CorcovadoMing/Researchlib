from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def DiracNet18(in_node, out_node, head=None, in_dim=3, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
    '''
    default_kwargs = dict(
        _op = op.DiracConv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        stem={'vgg': 1}, 
        type='vgg', 
        pool_freq=[5,9,13],
        filters=(64, -1),
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)

