from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def WideResNet28x10(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 12,
        type='wide-residual', 
        filters=(16, -1), 
        pool_freq=[5, 9], 
        preact=True
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)