from .utils import utils

# RL Dataset
from .environment import GymEnv, BSuiteEnv, GoogleFootBall

# Vision Dataset
from .np_dataset import _NumpyDataset
from .tf_dataset import _TFDataset
from .torch_dataset import _TorchDataset

# LFS Dataset
from .lfs import _LFS

# Tools
from .augmentations import Augmentations


class loader(object):
    utils = utils
    NumpyDataset = _NumpyDataset
    TorchDataset = _TorchDataset
    TFDataset = _TFDataset
    LFS = _LFS
