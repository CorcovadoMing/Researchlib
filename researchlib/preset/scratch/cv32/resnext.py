from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def ResNeXt50(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        stem={'vgg': 1}, 
        type='resnext-bottleneck', 
        pool_freq=[4,8,14],
        filters=(256, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)