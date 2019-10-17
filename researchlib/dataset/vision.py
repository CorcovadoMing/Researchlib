import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug
import random
from functools import partial
from .process_single import _process_single
from .preprocessing import preprocessing
from .augmentations import augmentations

class _VISION_GENERAL_LOADER:
    def __init__(self, is_train, ds, name, transpose):
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
        self.transpose = transpose
        
        
    def set_normalizer(self, type, mean, std):
        self.normalizer = preprocessing.set_normalizer(type, mean, std)
    
    
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
                                    normalizer=self.normalizer, augmentor=self.augmentor,
                                    transpose=self.transpose)
        ds = MultiProcessMapData(ds, 2, process_single_fn)
        ds = PrintData(ds)
        ds.reset_state()
        return ds
    
    def get_support_set(self, classes=[], shot=5):
        collect = {k: [] for k in classes}
        g = self.get_generator(1, epochs=3)
        count = 0
        for x, y in g:
            x = x[0]
            y = y[0]
            if y in classes and len(collect[y]) < shot:
                collect[y].append(x)
                count += 1
            if count == len(classes) * shot:
                break
        return np.stack(list(collect.values())), np.repeat(np.array(list(collect.keys())), shot).reshape(len(classes), shot)
    
    
    
def _CIFAR10(is_train=True, transpose=('NHWC', 'NCHW')):
    phase = 'train' if is_train else 'test'
    return _VISION_GENERAL_LOADER(is_train, dataset.Cifar10(phase, shuffle=is_train), name='cifar10', transpose=transpose)


def _NumpyDataset(x, y, is_train=True, name='', transpose=None):
    _inner_gen = DataFromList(list(zip(x, y)), shuffle=is_train)
    _inner_gen.data = x
    return _VISION_GENERAL_LOADER(is_train, _inner_gen, name=name, transpose=transpose)