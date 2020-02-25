from .utils import utils

# RL Dataset
from .environment import *

# Vision Dataset
from .np_dataset import _NumpyDataset
from .torch_dataset import _TorchDataset

# LFS Dataset
from .lfs import _LFS

# Tools
from .augmentations import Augmentations
from .preprocessing import Preprocessing


class loader(object):
    utils = utils
    NumpyDataset = _NumpyDataset
    TorchDataset = _TorchDataset
    LFS = _LFS
