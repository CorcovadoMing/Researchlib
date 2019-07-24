from .octconvblock import _OctConvBlock2d
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from .resblock import _ResBlock2d, _ResTransposeBlock2d
from .resnextblock import _ResNextBlock2d, _ResNextTransposeBlock2d
from .attentionblock import _AttentionBlock2d, _AttentionTransposeBlock2d
from .denseblock import _DenseBlock2d, _DenseTransposeBlock2d
from .basic_components import _CombinedDownSampling, _MaxPoolDownSampling, _InterpolateUpSampling, _ConvTransposeUpSampling


class downsampling(object):
    Combined = _CombinedDownSampling
    MaxPool = _MaxPoolDownSampling


class upsampling(object):
    Interpolate = _InterpolateUpSampling
    ConvTranspose = _ConvTransposeUpSampling


class block(object):
    # TODO: consistent API
    OctConvBlock2d = _OctConvBlock2d

    ConvBlock2d = _ConvBlock2d
    ConvTransposeBlock2d = _ConvTransposeBlock2d
    ResBlock2d = _ResBlock2d
    ResTransposeBlock2d = _ResTransposeBlock2d
    ResNextBlock2d = _ResNextBlock2d
    ResNextTransposeBlock2d = _ResNextTransposeBlock2d
    DenseBlock2d = _DenseBlock2d
    DenseTransposeBlock2d = _DenseTransposeBlock2d
    AttentionBlock2d = _AttentionBlock2d
    AttentionTransposeBlock2d = _AttentionTransposeBlock2d
