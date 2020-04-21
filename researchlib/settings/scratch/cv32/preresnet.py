from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn


def PreResNet18(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 8,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


def PreResNet34(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


def PreResNet50(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,8,14],
        filters=(256, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


def PreResNet101(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 33,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,8,31],
        filters=(256, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


def PreResNet152(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        _unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 50,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,12,48],
        filters=(256, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)