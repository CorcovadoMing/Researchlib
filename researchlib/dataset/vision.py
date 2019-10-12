import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug
import random
from functools import partial
from .process_single import _process_single
from .preprocessing import preprocessing
from .augmentations import augmentations

class _VISION_GENERAL_LOADER:
    def __init__(self, is_train, ds, name):
        self.is_train = is_train
        self.ds = ds
        self.name = name
        
        if is_train:
            # Record the statistic for normalizer 
            l = np.array([i[0] for i in self.ds.data])
            _VISION_GENERAL_LOADER.mean = np.mean(l, axis=tuple(range(l.ndim-1))) 
            _VISION_GENERAL_LOADER.std = np.std(l, axis=tuple(range(l.ndim-1)))
            del l
        
        self.normalizer = []
        self.augmentor = []
        self.include_y = False
        
        
    def _set_normalizer(self, local=False):
        if local:
            pass
        else:
            self.normalizer = [partial(preprocessing.normalize, 
                                       mean = _VISION_GENERAL_LOADER.mean, 
                                       std = _VISION_GENERAL_LOADER.std)]
    
    
    def _set_augmentor(self, augmentor, include_y=False):
        mapping = {
            'hflip': augmentations.HFlip(),
            'crop': augmentations.Crop(32, 32, 4)
        }
        
        self.augmentor = []
        for i in augmentor:
            self.augmentor.append(mapping[i])
        self.include_y = include_y
        
        
    def get_generator(self, batch_size=512, **kwargs):
        ds = BatchData(self.ds, batch_size, remainder=True)
        process_single_fn = partial(_process_single, 
                                    is_train=self.is_train, include_y=self.include_y, 
                                    normalizer=self.normalizer, augmentor=self.augmentor)
        ds = MultiProcessMapDataZMQ(ds, 4, process_single_fn)
        if self.is_train:
            ds = MultiProcessPrefetchData(ds, 2048//batch_size, 4)
        ds = PrintData(ds)
        ds.reset_state()
        return ds
    
    
    
def _CIFAR10(is_train=True):
    phase = 'train' if is_train else 'test'
    return _VISION_GENERAL_LOADER(is_train, dataset.Cifar10(phase, shuffle=is_train), name='cifar10')


def _NumpyDataset(x, y, is_train=True, name=''):
    _inner_gen = DataFromList(list(zip(x, y)), shuffle=is_train)
    _inner_gen.data = x
    return _VISION_GENERAL_LOADER(is_train, _inner_gen, name=name)