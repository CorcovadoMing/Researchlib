from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def Dawnfast(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 3,
        stem={'whitening': 1},
        type={'order':['dawn', 'vgg'], 'type':'alternative'}, 
        filters=(64, 512), 
        act_type='celu', 
        prepool=True, 
        freeze_scale=True,
        norm_type=op.GhostBatchNorm2d
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
        model.append(op.Multiply(1/2))
    return Node(out_node, nn.Sequential(*model), in_node)