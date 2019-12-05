from .imagenet import ImageNet
from .scratch import Scratch
from .loader import Loader
from .config import Config


class Preset(object):
    ImageNet = ImageNet
    Scratch = Scratch
    Config = Config
    Loader = Loader