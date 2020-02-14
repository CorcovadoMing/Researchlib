from tensorpack.dataflow import *
from functools import partial
from ...dataset.process_random_augment import _process_random_augment
from ...dataset import Augmentations
from torch import nn


class _RandAugment(nn.Module):
    def __init__(self, N=2, M=1, include_y = False):
        super().__init__()
        self.augmentor = [
            Augmentations.CircularCrop(32, 32, 8), 
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(32, 32, 8),
            Augmentations.Invert(),
            Augmentations.Equalize(),
            Augmentations.Identical(),
            Augmentations.ShearX(),
            Augmentations.ShearY(),
            Augmentations.TranslateX(),
            Augmentations.TranslateY(),
            Augmentations.Rotate(),
            Augmentations.Solarize(),
            Augmentations.Posterize(),
            Augmentations.Contrast(),
            Augmentations.Color(),
            Augmentations.Brightness(),
            Augmentations.Sharpness()
        ]
        self.N = N
        self.M = M
        self.include_y = include_y
        self.phase = 0
        self.train_ds = None
        self.val_ds = None
        
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
            process_augment_fn = partial(
                _process_random_augment,
                N = self.N,
                M = self.M,
                include_y = self.include_y,
                augmentor = self.augmentor
            )
            self.train_ds = MapData(ds, process_augment_fn)
        
        if self.val_ds is None and self.phase == 1:
            self.val_ds = ds
        
        if self.phase == 0:
            return self.train_ds
        elif self.phase == 1:
            return self.val_ds
        else:
            return -1 # Custom predict phase
