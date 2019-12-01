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
from .preprocessing import preprocessing


def rename_fn(dp):
    return dp['image'], dp['label']

class _TFDataset:
    def __init__(self, name, is_train, transpose = ('NHWC', 'NCHW')):
        self.is_train = is_train
        self.name = name

        split = tfds.Split.TRAIN if is_train else tfds.Split.TEST
        phase = 'train' if is_train else 'test'
        ds = tfds.builder(name)
        info = ds.info
        self.length = info.splits[phase].num_examples
        ds.download_and_prepare()
        self.ds = ds.as_dataset(split = split, shuffle_files = False)
        self.transpose = transpose

    def get_generator(self, batch_size = 512, **kwargs):
        # TODO: support the fixed size iteration
        ds = self.ds.shuffle(10240).repeat(kwargs['epochs']).batch(batch_size).prefetch(2048)
        ds = DataFromGenerator(tfds.as_numpy(ds))
        ds.__len__ = lambda: math.ceil(self.length / batch_size)
        ds = MultiProcessMapDataZMQ(ds, 2, rename_fn, buffer_size = 8)
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
