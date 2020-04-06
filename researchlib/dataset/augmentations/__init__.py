from .crop import CircularCrop
from .hflip import HFlip
from .cutout import Cutout
from .autocontrast import AutoContrast
from .invert import Invert
from .equalize import Equalize
from .identical import Identical
from .shearx import ShearX
from .sheary import ShearY
from .translatex import TranslateX
from .translatey import TranslateY
from .rotate import Rotate
from .solarize import Solarize
from .posterize import Posterize
from .contrast import Contrast
from .color import Color
from .brightness import Brightness
from .sharpness import Sharpness
from .svd_blur import SVDBlur
from .sparkle import Sparkle
from .scale import Scale


class Augmentations(object):
    CircularCrop = CircularCrop
    HFlip = HFlip
    Cutout = Cutout
    AutoContrast = AutoContrast
    Invert = Invert
    Equalize = Equalize
    Identical = Identical
    ShearX = ShearX
    ShearY = ShearY
    TranslateX = TranslateX
    TranslateY = TranslateY
    Rotate = Rotate
    Solarize = Solarize
    Posterize = Posterize
    Contrast = Contrast
    Color = Color
    Brightness = Brightness
    Sharpness = Sharpness
    SVDBlur = SVDBlur
    Sparkle = Sparkle
    Scale = Scale