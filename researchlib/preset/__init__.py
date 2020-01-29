from .pretrained import Pretrained
from .scratch import Scratch
from .dataset import Dataset
from .config import Config
from .application import Application

class Preset(object):
    Pretrained = Pretrained
    Scratch = Scratch
    Config = Config
    Dataset = Dataset
    Application = Application