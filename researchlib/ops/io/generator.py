from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop
from tensorpack.dataflow import *
from torch import nn

class _Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.fp16 = False
        self.phase = 0
        
    def prepare_state(self, fp16):
        self.fp16 = fp16
    
    def set_phase(self, phase):
        self.phase = phase
    
    def reset_parameters(self):
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
            ds = PrintData(ds)
            ds.reset_state()
            self.train_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.val_ds is None and self.phase == 1:
            ds = PrintData(ds)
            ds.reset_state()
            self.val_ds = BackgroundGenerator(inifinity_loop(ds), fp16=self.fp16)
            
        if self.phase == 0:
            return next(iter(self.train_ds))
        elif self.phase == 1:
            return next(iter(self.val_ds))