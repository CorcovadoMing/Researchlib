from ...models import AutoConvNet, Heads, Node
from ...ops import op
from ...blocks import unit
from torch import nn


def RandWire(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.SepConv2d,
        unit = unit.Conv,
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



def Dawnfast(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
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


def RevNet110(in_node, out_node, head=None, in_dim=3, **kwargs):
    # Need to be verified
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
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
        


def PreResNet18(in_node, out_node, head=None, in_dim=3, **kwargs):
    default_kwargs = dict(
        _op = op.Conv2d,
        unit = unit.Conv,
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
        unit = unit.Conv,
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
        unit = unit.Conv,
        input_dim = in_dim,
        total_blocks = 16,
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    model = [AutoConvNet(**default_kwargs)]
    if head != None:
        model.append(Heads(head, reduce_type='avg', channels_transform=True, preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


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
        model.append(Heads(head, reduce_type='avg', preact=True))
    return Node(out_node, nn.Sequential(*model), in_node)


class CV32x32(object):
    RandWire = RandWire
    
    Dawnfast = Dawnfast
    
    ResNet18 = ResNet18
    ResNet110 = ResNet110
    
    RevNet110 = RevNet110
    
    PreResNet18 = PreResNet18
    PreResNet34 = PreResNet34
    PreResNet50 = PreResNet50
    
    WideResNet28x10 = WideResNet28x10
    PyramidNet272 = PyramidNet272
    