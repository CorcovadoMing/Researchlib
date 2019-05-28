from .octconvblock import _OctConvBlock2d
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
from .resblock import _ResBlock2d, _ResTransposeBlock2d

class block(object):
    OctConvBlock2d=_OctConvBlock2d
    ConvBlock2d=_ConvBlock2d
    ConvTransposeBlock2d=_ConvTransposeBlock2d
    ResBlock2d=_ResBlock2d
    ResTransposeBlock2d=_ResTransposeBlock2d