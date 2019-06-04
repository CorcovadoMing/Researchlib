from .octconvblock import _OctConvBlock2d
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from .resblock import _ResBlock2d, _ResTransposeBlock2d
from .resnextblock import _ResNextBlock2d, _ResNextTransposeBlock2d
from .attentionblock import _AttentionBlock
from .denseblock import _DenseBlock2d, _DenseTransposeBlock2d

class block(object):
    OctConvBlock2d=_OctConvBlock2d
    ConvBlock2d=_ConvBlock2d
    ConvTransposeBlock2d=_ConvTransposeBlock2d
    ResBlock2d=_ResBlock2d
    ResTransposeBlock2d=_ResTransposeBlock2d
    ResNextBlock2d=_ResNextBlock2d
    ResNextTransposeBlock2d=_ResNextTransposeBlock2d
    DenseBlock2d=_DenseBlock2d
    DenseTransposeBlock2d=_DenseTransposeBlock2d
    
    AttentionBlock=_AttentionBlock