from .pretrained import Pretrained
from .scratch import Scratch
from .dataset import Dataset
from .training import Training
from .application import Application

class Preset(object):
    Pretrained = Pretrained
    Scratch = Scratch
    Training = Training
    Dataset = Dataset
    Application = Application