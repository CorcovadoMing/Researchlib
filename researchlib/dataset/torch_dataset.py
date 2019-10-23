import torch
import torchvision
import numpy as np
from functools import partial
from tensorpack.dataflow import *
from .process_single import _process_single
from .augmentations import augmentations
from .preprocessing import preprocessing
from .vision import _NumpyDataset


def _TorchDataset(name, is_train, transpose = ('NHWC', 'NCHW')):
    dataset_fn = None
    for i in torchvision.datasets.__dict__:
        if i.lower() == name:
            dataset_fn = torchvision.datasets.__dict__[i]
            break

    if dataset_fn is None:
        raise ValueError(f'No dataset {name} founded')

    ds = dataset_fn(train = is_train, download = True, root = './data')
    return _NumpyDataset(ds.data, ds.targets, is_train, name = name, transpose = transpose)