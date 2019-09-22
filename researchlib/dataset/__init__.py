# Dataset Wrapper
from .from_numpy import FromNumpy
from .from_vision_dataset import VisionDataset
from .environment import GymEnv

# Dataset
from .vision import _MNIST, _FashionMNIST, _KMNIST, _CIFAR10

class vision(object):
    MNIST = _MNIST
    KMNIST = _KMNIST
    FashionMNIST = _FashionMNIST
    CIFAR10 = _CIFAR10
