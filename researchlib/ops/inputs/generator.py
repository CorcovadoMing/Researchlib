from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop
from tensorpack.dataflow import *
from torch import nn
import random
import numpy as np
from functools import partial
from ...dataset import Augmentations, Preprocessing

def _processing_function(
    dp,
    include_y,
    augmentor,
    normalizer,
    N,
    M,
    data_key = 0,
    label_key = 1,
):
    x = dp[data_key]
    y = dp[label_key]
    x_org = x.astype(np.float32)

    # Augmentation
    augmentor = augmentor if len(augmentor) < N else random.choices(augmentor, k=N)
    for op in augmentor:
        options = op.options()
        x = op(x, **random.choice(options))

    if x.shape[:-1] == y.shape[:-1]:
        do_y = True
    else:
        do_y = False

    x = x.astype(np.float32)
    y = y.astype(np.float32) if do_y else y.astype(np.int64)

    # Normalization
    for op in normalizer:
        x = op(x)
        x_org = op(x_org)
        if do_y:
            y = op(y)
            
    return x, y, x_org


def _flat_list(l):
    s = [i if type(i) == list else [i] for i in l]
    return sum(s, [])


class _Generator(nn.Module):
    def __init__(self, 
                 *preprocessing_list, 
                 include_y = False, 
                 N = 2, 
                 M = 1):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.fp16 = False
        self.phase = 0
        self.N = N
        self.M = M
        self.include_y = include_y
        
        preprocessing_list = _flat_list(preprocessing_list)
        normalize_list = []
        augment_list = []
        for i in preprocessing_list:
            try:
                if 'options' in i.__class__.__dict__:
                    augment_list.append(i)
                else:
                    normalize_list.append(i)
            except:
                normalize_list.append(i)
        
        self.train_processing_function = partial(
            _processing_function,
            N = self.N,
            M = self.M,
            include_y = False,
            normalizer = normalize_list,
            augmentor = augment_list
        )
        
        self.val_processing_function = partial(
            _processing_function,
            N = self.N,
            M = self.M,
            include_y = False,
            normalizer = normalize_list,
            augmentor = []
        )
        
    def prepare_state(self, fp16, batch_size):
        self.fp16 = fp16
        self.batch_size = batch_size
    
    def set_phase(self, phase):
        self.phase = phase
    
    def clear_source(self, is_train):
        try:
            del self.train_ds
        except:
            pass
        
        try:
            del self.val_ds
        except:
            pass
            
        self.train_ds = None
        self.val_ds = None
    
    def forward(self, ds):
        if self.train_ds is None and self.phase == 0:
            ds = MapData(ds, self.train_processing_function)
            ds = BatchData(ds, self.batch_size, remainder = True)
            ds = PrintData(ds)
            ds.reset_state()
            self.train_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.val_ds is None and self.phase == 1:
            ds = MapData(ds, self.val_processing_function)
            ds = BatchData(ds, self.batch_size, remainder = True)
            ds = PrintData(ds)
            ds.reset_state()
            self.val_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.phase == 0:
            return next(iter(self.train_ds))
        elif self.phase == 1:
            return next(iter(self.val_ds))
        else:
            return -1 # Custom predict phase