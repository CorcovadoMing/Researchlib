from .octconvblock import _OctConvBlock2d
from .convblock import _ConvBlock2d, _ConvTransposeBlock2d

class block(object):
    OctConvBlock2d=_OctConvBlock2d
    ConvBlock2d=_ConvBlock2d
    ConvTransposeBlock2d=_ConvTransposeBlock2d