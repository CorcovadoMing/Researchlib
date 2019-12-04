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
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1),
        preact=True
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 8, **default_kwargs)


def PreResNet110(**kwargs):
    # TODO: Verify
    default_kwargs = dict(
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[19,37],
        preact=True,
        shortcut_norm=True,
        filters=(32, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 54, **default_kwargs)


def PreRevNet110(**kwargs):
    # TODO: Verify
    default_kwargs = dict(
        stem={'vgg': 1}, 
        type='rev-residual', 
        pool_freq=[10,19],
        preact=True,
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 27, **default_kwargs)


def PreResNet18(**kwargs):
    default_kwargs = dict(
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[3,5,7],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 8, **default_kwargs)


def PreResNet34(**kwargs):
    default_kwargs = dict(
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 16, **default_kwargs)


def PreResNet50(**kwargs):
    default_kwargs = dict(
        preact=True,
        preact_bn_shared=True,
        stem={'vgg': 1}, 
        type='residual-bottleneck', 
        pool_freq=[4,8,14],
        filters=(64, -1)
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 16, **default_kwargs)


def WideResNet28x10(**kwargs):
    default_kwargs = dict(
        type='wide-residual', 
        filters=(16, -1), 
        pool_freq=[5, 9], 
        preact=True
    )
    default_kwargs.update(kwargs)
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 12, **default_kwargs)


def PyramidNet272(**kwargs):
    default_kwargs = dict(
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
    return AutoConvNet(op.Conv2d, unit.Conv, 3, 90, **default_kwargs)


class Template(object):
    Dawnfast = Dawnfast
    ResNet18 = ResNet18
    PreResNet110 = PreResNet110
    PreRevNet110 = PreRevNet110
    PreResNet18 = PreResNet18
    PreResNet50 = PreResNet50
    WideResNet28x10 = WideResNet28x10
    PyramidNet272 = PyramidNet272