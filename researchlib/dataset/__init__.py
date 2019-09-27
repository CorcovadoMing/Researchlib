# RL Dataset
from .environment import GymEnv

# Vision Dataset
from .vision import _CIFAR10, _NumpyDataset
from .tf_dataset import _TFDataset
from .torch_dataset import _TorchDataset

class vision(object):
    CIFAR10 = _CIFAR10
    NumpyDataset = _NumpyDataset

class general(object):
    TorchDataset = _TorchDataset
    
class thirdparty(object):
    TFDataset = _TFDataset