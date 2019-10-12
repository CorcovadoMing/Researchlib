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
        
        
    def _set_normalizer(self, local=False):
        if not local:
            print("TensorFlow dataset didn't support the global initialization yet, \
            consider to use local (per minibatch) normalization if needs")
            self.normalizer = []
        else:
            self.normalizer = []
    
    
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