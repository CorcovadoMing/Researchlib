from .crop import Crop, CircularCrop
from .hflip import HFlip
from .cutout import Cutout
from .nonlinear_jitter import NonlinearJitter
from .elastic_transform import ElasticTransform

class Augmentations(object):
    Crop = Crop
    CircularCrop = CircularCrop
    HFlip = HFlip
    Cutout = Cutout
    NonlinearJitter = NonlinearJitter
    ElasticTransform = ElasticTransform
    