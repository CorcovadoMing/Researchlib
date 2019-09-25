import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_datasets as tfds
import tensorflow as tf
tf.enable_eager_execution()
from tensorpack.dataflow import *
import numpy as np
import math


class _TFDataset:
    def __init__(self, name, is_train):
        self.is_train = is_train
        
        split = tfds.Split.TRAIN if is_train else tfds.Split.TEST
        phase = 'train' if is_train else 'test'
        ds = tfds.builder(name)
        info = ds.info
        self.length = info.splits[phase].num_examples
        ds.download_and_prepare()
        self.ds = ds.as_dataset(split=split, shuffle_files=False)
        
        self.normalizer = []
        self.augmentor = []
        
        
    def _set_normalizer(self):
        self.normalizer = []
    
    
    def _set_augmentor(self, augmentor):
        mapping = {
            'hflip': imgaug.Flip(horiz=True),
            'crop': imgaug.RandomApplyAug(
                        imgaug.AugmentorList([
                            imgaug.CenterPaste((40, 40)),
                            imgaug.RandomCrop((32, 32)),
                        ]),
                    0.5),
            'cutout': imgaug.RandomApplyAug(imgaug.RandomCutout(8, 8), 0.5)
        }
        
        _aug = []
        for i in augmentor:
            _aug.append(mapping[i])
        self.augmentor = [imgaug.RandomOrderAug(_aug)]
    
    
    def get_generator(self, batch_size=512, **kwargs):
        
        ds = self.ds.shuffle(10240).repeat(kwargs['epochs']).batch(batch_size).prefetch(2048)
        ds = DataFromGenerator(tfds.as_numpy(ds))
        ds.__len__ = lambda: math.ceil(self.length / batch_size)
        
        def batch_mapf(dp):
            x = dp['image'].copy()
            y = dp['label']
            
            if self.is_train:
                for i in range(len(x)):
                    for op in self.augmentor:
                        x[i] = op.augment(x[i])
            
            for op in self.normalizer:
                x = op.augment(x)
            
            return np.moveaxis(x, -1, 1).astype(np.float32), np.array(y).astype(np.int64)
        
        
        ds = MapData(ds, batch_mapf)
        ds = PrintData(ds)
        ds.reset_state()
        return ds