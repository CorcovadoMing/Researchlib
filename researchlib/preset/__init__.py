from .imagenet import ImageNet
from .scratch import Scratch
from .dataset import Dataset
from .config import Config


class Preset(object):
    ImageNet = ImageNet
    Scratch = Scratch
    Config = Config
    Dataset = Dataset