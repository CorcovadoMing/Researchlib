import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug
import random
from functools import partial
from .process_single import _process_single
from .process_augment import _process_augment
from .preprocessing import preprocessing


class _VISION_GENERAL_LOADER:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name

    def get_generator(self, batch_size = 512, **kwargs):
        ds = self.ds
        if 'fixed_batch' in kwargs:
            ds = FixedSizeData(ds, batch_size * kwargs['fixed_batch'], keep_state=False)
            ds = LocallyShuffleData(ds, batch_size * kwargs['fixed_batch'])
        ds = BatchData(ds, batch_size, remainder = True)
        return ds

    def get_support_set(self, classes = [], shot = 5):
        collect = {k: [] for k in classes}
        g = self.get_generator(1, epochs = 3)
        count = 0
        for x, y in g:
            x = x[0]
            y = y[0]
            if y in classes and len(collect[y]) < shot:
                collect[y].append(x)
                count += 1
            if count == len(classes) * shot:
                break
        return np.stack(list(collect.values())), np.repeat(np.array(list(collect.keys())),
                                                           shot).reshape(len(classes), shot)


def _CIFAR10(is_train = True, shuffle = True):
    phase = 'train' if is_train else 'test'
    return _VISION_GENERAL_LOADER(
        dataset.Cifar10(phase, shuffle = shuffle),
        name = 'cifar10',
    )


def _NumpyDataset(x, y, shuffle = True, name = ''):
    _inner_gen = DataFromList(list(zip(x, y)), shuffle = shuffle)
    _inner_gen.data = x
    return _VISION_GENERAL_LOADER(_inner_gen, name = name)
