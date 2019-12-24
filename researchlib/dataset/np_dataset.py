from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader


def _NumpyDataset(x, y, shuffle = True, name = '', size = None):
    _inner_gen = DataFromList(list(zip(x, y)), shuffle = shuffle)
    _inner_gen.data = x
    return _GeneralLoader(_inner_gen, name = name, size = size)