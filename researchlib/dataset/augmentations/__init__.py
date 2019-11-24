from .crop import Crop
from .hflip import HFlip
from .cutout import Cutout


class Augmentations(object):
    Crop = Crop
    HFlip = HFlip
    Cutout = Cutout
    