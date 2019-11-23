from ..models import AutoConvNet
from ..ops import op
from ..blocks import unit


def Dawnfast(**kwargs):
    default_kwargs = dict(
        stem={'whitening': 1},
        type={'order':['dawn', 'vgg'], 'type':'alternative'}, 
        filters=(64, 512), 
        act_type='celu', 
        prepool=True, 
        freeze_scale=True,
        norm_type=op.GhostBatchNorm2d
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 3, **default_kwargs)


def ResNet18(**kwargs):
    default_kwargs = dict(
        stem={'whitening': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 8, **default_kwargs)


def PreResNet18(**kwargs):
    default_kwargs = dict(
        preact=True,
        preact_bn_shared=True,
        stem={'whitening': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 8, **default_kwargs)


def WideResNet28x10(**kwargs):
    default_kwargs = dict(
        type='wide-residual', 
        filters=(16, -1), 
        pool_freq=[5, 9], 
        preact=True
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 12, **default_kwargs)


class Template(object):
    Dawnfast = Dawnfast
    ResNet18 = ResNet18
    PreResNet18 = PreResNet18
    WideResNet28x10 = WideResNet28x10