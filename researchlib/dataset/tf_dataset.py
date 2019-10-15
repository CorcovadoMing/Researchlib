import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorpack.dataflow import *
import numpy as np
import math
from functools import partial
from .process_single import _process_single
from .augmentations import augmentations
from .preprocessing import preprocessing


class _TFDataset:
    def __init__(self, name, is_train):
        self.is_train = is_train
        self.name = name
        
        split = tfds.Split.TRAIN if is_train else tfds.Split.TEST
        phase = 'train' if is_train else 'test'
        ds = tfds.builder(name)
        info = ds.info
        self.length = info.splits[phase].num_examples
        ds.download_and_prepare()
        self.ds = ds.as_dataset(split=split, shuffle_files=False)
        
        self.normalizer = []
        self.augmentor = []
        self.include_y = False
        
        
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
        ds = self.ds.shuffle(10240).repeat(kwargs['epochs']).batch(batch_size).prefetch(2048)
        ds = DataFromGenerator(tfds.as_numpy(ds))
        ds.__len__ = lambda: math.ceil(self.length / batch_size)
        process_single_fn = partial(_process_single, 
                                    is_train=self.is_train, include_y=self.include_y, 
                                    normalizer=self.normalizer, augmentor=self.augmentor, 
                                    data_key='image', label_key='label')
        ds = MultiProcessMapDataZMQ(ds, 4, process_single_fn)
        ds = PrintData(ds)
        ds.reset_state()
        return ds
    
    def get_support_set(self, classes=[], shot=5):
        data, label = [], []
        collect = {k: 0 for k in classes}
        g = self.get_generator(1, epochs=3)
        for x, y in g:
            x = x[0]
            y = y[0]
            if y in classes and collect[y] < shot:
                data.append(x)
                label.append(y)
                collect[y] += 1
            if sum(collect.values()) == len(classes) * shot:
                break
        return np.array(data), np.array(label)