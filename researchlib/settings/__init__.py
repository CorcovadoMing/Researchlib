from .pretrained import Pretrained
from .scratch import Scratch
from .dataset import Dataset
from .training import Training
from .application import Application

from .set_storage import SetStorage


class Settings(object):
    Pretrained = Pretrained
    Scratch = Scratch
    Training = Training
    Dataset = Dataset
    Application = Application
    SetStorage = SetStorage