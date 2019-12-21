from .pretrained import Pretrained
from .scratch import Scratch
from .dataset import Dataset
from .config import Config


class Preset(object):
    Pretrained = Pretrained
    Scratch = Scratch
    Config = Config
    Dataset = Dataset