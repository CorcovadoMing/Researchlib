from tensorpack.dataflow import *
from functools import partial
from ...dataset.process_augment import _process_augment
from ...dataset.preprocessing import preprocessing
from torch import nn


class _Augmentation(nn.Module):
    def __init__(self, augmentor, include_y = False):
        super().__init__()
        self.augmentor = augmentor
        self.include_y = include_y
        self.phase = 0
        self.train_ds = None
        self.val_ds = None
        
    def set_phase(self, phase):
        self.phase = phase
        
    def forward(self, ds):
        if self.train_ds is None and self.phase == 0:
            process_augment_fn = partial(
                _process_augment,
                include_y = self.include_y,
                augmentor = self.augmentor
            )
            self.train_ds = MultiProcessMapDataZMQ(ds, 2, process_augment_fn, buffer_size = 8)
        
        if self.val_ds is None and self.phase == 1:
            self.val_ds = ds
        
        if self.phase == 0:
            return self.train_ds
        elif self.phase == 1:
            return self.val_ds
