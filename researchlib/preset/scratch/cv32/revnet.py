from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def RevNet110(in_node, out_node, head=None, in_dim=3, **kwargs):
    # Need to be verified
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 27,
        stem={'vgg': 1}, 
        type='rev-residual', 
        pool_freq=[10,19],
        erased_act=True,
        filters=(32, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)
        