from .utils import utils

# RL Dataset
from .environment import GymEnv

# Vision Dataset
from .tp_dataset import _CIFAR10, _NumpyDataset
from .tf_dataset import _TFDataset
from .torch_dataset import _TorchDataset

# Tools
from .augmentations import Augmentations


class loader(object):
    utils = utils
    CIFAR10 = _CIFAR10
    NumpyDataset = _NumpyDataset
    TorchDataset = _TorchDataset
    TFDataset = _TFDataset
