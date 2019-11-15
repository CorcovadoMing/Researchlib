from tensorpack.dataflow import *
from functools import partial
from ...dataset.process_single import _process_single
from ...dataset.preprocessing import preprocessing
from torch import nn


class _Normalize(nn.Module):
    def __init__(self, type, mean, std):
        super().__init__()
        self.normalizer = preprocessing.set_normalizer(type, mean, std)
        self.phase = 0
        self.train_ds = None
        self.val_ds = None
        
    def set_phase(self, phase):
        self.phase = phase
        
    def forward(self, ds):
        if self.train_ds is None and self.phase == 0:
            process_single_fn = partial(
                _process_single,
                normalizer = self.normalizer
            )
            self.train_ds = MultiProcessMapDataZMQ(ds, 2, process_single_fn, buffer_size = 8)
        
        if self.val_ds is None and self.phase == 1:
            process_single_fn = partial(
                _process_single,
                normalizer = self.normalizer
            )
            self.val_ds = MultiProcessMapDataZMQ(ds, 2, process_single_fn, buffer_size = 8)
        
        if self.phase == 0:
            return self.train_ds
        elif self.phase == 1:
            return self.val_ds