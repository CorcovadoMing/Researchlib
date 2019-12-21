from ...models import AutoConvNet, Heads, Node
from ...ops import op
from ...blocks import unit
from torch import nn
import math


def ResNet(size, pool_freq, in_node, out_node, head=None, start_filter=64, end_filter=-1, in_dim=3, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
    '''
    # original blocks = pool_freq
    # ... body ...
    # final 8x8 size
    total_blocks = int((math.log2(size)-3+1) * pool_freq)
    _pool_freq = [i+pool_freq+1 for i in range(0, total_blocks, pool_freq)]
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = total_blocks,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=_pool_freq,
        filters=(start_filter, end_filter),
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)



class CVGeneric(object):
    ResNet = ResNet
    
