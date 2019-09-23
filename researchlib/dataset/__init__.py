# RL Dataset
from .environment import GymEnv

# Vision Dataset
from .vision import _CIFAR10, _NumpyDataset

class vision(object):
    CIFAR10 = _CIFAR10
    NumpyDataset = _NumpyDataset
