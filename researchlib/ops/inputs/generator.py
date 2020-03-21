from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop
from tensorpack.dataflow import *
from torch import nn
import random
import copy
import numpy as np
from functools import partial
from ...dataset import Augmentations, Preprocessing


def _processing_function(
    dp,
    augmentor,
    normalizer,
    N,
    M,
):
    x_batch = dp[0].copy()
    y_batch = dp[1].copy()
    x_org_batch = copy.deepcopy(x_batch).astype(np.float32)
    
    new_x_batch = []
    new_y_batch = []
    new_x_org_batch = []
    
    for x, y, x_org in zip(x_batch, y_batch, x_org_batch):
        # Augmentation
        this_augmentor = augmentor if len(augmentor) < N else np.random.choice(augmentor, N, replace=False)
        for op in this_augmentor:
            options = op.options(prob=1)
            x = op(x, **np.random.choice(options))

        if x.shape[:-1] == y.shape[:-1]:
            do_y = True
        else:
            do_y = False

        x = x.astype(np.float32)
        y = y.astype(np.float32) if do_y else y.astype(np.int64)

        # Fix grayscale
        if x.ndim == 2:
            x = x[..., None]
            if y.ndim == 2 and do_y:
                y = y[..., None]

        # Normalization
        for op in normalizer:
            x = op(x)
            x_org = op(x_org)
            if do_y:
                y = op(y)
                
        new_x_batch.append(x)
        new_y_batch.append(y)
        new_x_org_batch.append(x_org)

    new_x_batch = np.array(new_x_batch)
    new_y_batch = np.array(new_y_batch)
    new_x_org_batch = np.array(new_x_org_batch)
    
    return [new_x_batch, new_y_batch, new_x_org_batch] + dp[2:]


def _flat_list(l):
    s = [i if type(i) == list else [i] for i in l]
    return sum(s, [])


class _Generator(nn.Module):
    '''
        Set worker > 0 for multithreading
        It may have problem if the total dataset is smaller than batch_size * buffer_size
        Consider to set the worker = 0 for any problem
    '''
    def __init__(self, 
                 *preprocessing_list, 
                 worker = 8,
                 buffer = 2,
                 N = 2, 
                 M = 1):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.fp16 = False
        self.data_info = True
        self.phase = 0
        self.N = N
        self.M = M
        self.worker = worker
        self.buffer = buffer
        
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
            normalizer = normalize_list,
            augmentor = augment_list
        )
        
        self.val_processing_function = partial(
            _processing_function,
            N = self.N,
            M = self.M,
            normalizer = normalize_list,
            augmentor = []
        )
        
    def prepare_state(self, fp16, batch_size, data_info=True):
        self.fp16 = fp16
        self.batch_size = batch_size
        self.data_info = data_info
    
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
            ds = BatchData(ds, self.batch_size, remainder = True)
            if self.worker > 0:
                ds = MultiProcessMapData(ds, self.worker, self.train_processing_function, self.buffer, strict=True)
            else:
                ds = MapData(ds, self.train_processing_function)
            if self.data_info:
                ds = PrintData(ds)
            ds.reset_state()
            self.train_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.val_ds is None and self.phase == 1:
            ds = BatchData(ds, self.batch_size, remainder = True)
            ds = MapData(ds, self.val_processing_function)
            if self.data_info:
                ds = PrintData(ds)
            ds.reset_state()
            self.val_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.phase == 0:
            return next(iter(self.train_ds))
        elif self.phase == 1:
            return next(iter(self.val_ds))
        else:
            return -1 # Custom predict phase