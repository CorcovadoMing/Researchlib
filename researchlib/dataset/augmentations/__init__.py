from .crop import Crop
from .hflip import HFlip
from .cutout import Cutout
from .nonlinear_jitter import NonlinearJitter
from .elastic_transform import ElasticTransform

class Augmentations(object):
    Crop = Crop
    HFlip = HFlip
    Cutout = Cutout
    NonlinearJitter = NonlinearJitter
    ElasticTransform = ElasticTransform
    