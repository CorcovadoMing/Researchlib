from .crop import CircularCrop
from .hflip import HFlip
from .cutout import Cutout
from .autocontrast import AutoContrast
from .invert import Invert
from .equalize import Equalize


class Augmentations(object):
    CircularCrop = CircularCrop
    HFlip = HFlip
    Cutout = Cutout
    AutoContrast = AutoContrast
    Invert = Invert
    Equalize = Equalize
    