# Dataset Wrapper
from .from_numpy import FromNumpy
from .from_hdf5 import FromHDF5
from .from_vision_dataset import VisionDataset
from .from_text_dataset import TextDataset
from .dali import FromDali
from .environment import GymEnv

# Dataset
from .vision import _MNIST, _FashionMNIST, _KMNIST, _CIFAR10
from .text import _AGNews

# Augmentation
from .augmentator import Augmentator
from .ts_aug_soli import *

#from .pointer_network import *

class text(object):
    AGNews=_AGNews
    
class vision(object):
    MNIST=_MNIST
    KMNIST=_KMNIST
    FashionMNIST=_FashionMNIST
    CIFAR10=_CIFAR10
