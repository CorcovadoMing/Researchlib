# RL Dataset
from .environment import GymEnv

# Vision Dataset
from .vision import _CIFAR10, _NumpyDataset
from .tf_dataset import _TFDataset
from .torch_dataset import _TorchDataset


class loader(object):
    CIFAR10 = _CIFAR10
    NumpyDataset = _NumpyDataset
    TorchDataset = _TorchDataset
    TFDataset = _TFDataset
