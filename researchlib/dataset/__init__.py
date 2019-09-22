# Dataset Wrapper
from .from_numpy import FromNumpy
from .environment import GymEnv

# Dataset
from .vision import _CIFAR10

class vision(object):
    CIFAR10 = _CIFAR10
