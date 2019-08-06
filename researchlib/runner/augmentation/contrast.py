from ..template import template
from ...utils import mapping
from .grayscale import _grayscale
import numpy as np
import torch
import random


class Contrast(template.TorchAugmentation):
    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag
    
    def _aug_fn(self, img, var):
        gs = _grayscale(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, var)
        return img.lerp(gs, alpha)
    
    def forward_single(self, x, y, mag):
        x = [ self._aug_fn(i, mag) for i in x]
        if self.include_y:
            y = [ self._aug_fn(i, mag) for i in y]
        return x, y
    
    
    
    class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)