from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def ResNet18(in_node, out_node, head=None, in_dim=3, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
    '''
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 8,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1),
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)


def ResNet34(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)


def ResNet50(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)


def ResNet110(in_node, out_node, head=None, in_dim=3, **kwargs):
    '''
        Deep Residual Learning for Image Recognition
    '''
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 54,
        stem={'vgg': 1}, 
        type='residual',
        shortcut='padding',
        pool_freq=[19,37],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg'))
    return Node(out_node, nn.Sequential(*model), in_node)