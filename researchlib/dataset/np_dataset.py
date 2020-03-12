from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader


def _NumpyDataset(*data, shuffle = True, name = ''):
    _inner_gen = DataFromList(list(zip(*data)), shuffle = shuffle)
    _inner_gen.data = data[0]
    return _GeneralLoader(_inner_gen, name = name)