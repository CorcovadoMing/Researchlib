from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def RandWire(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.SepConv2d,
        _unit = unit.Conv,
        input_dim = 3,
        total_blocks = 3,
        stem={'vgg': 1},
        type='randwire',
        pool_freq=[2,3],
        filters=(64, -1),
        preact=True,
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True))
    return Node(out_node, nn.Sequential(*model), in_node)

