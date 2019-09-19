from ..template import template
from ...utils import mapping
import numpy as np
import torch
import random


class Brightness(template.TorchAugmentation):
    def __init__(self, prob = None, mag = None, include_y = False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag

    def _aug_fn(self, img, var):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, var)
        return img.lerp(gs, alpha)

    def forward_single(self, x, y, mag):
        x = [self._aug_fn(i, mag) for i in x]
        if self.include_y:
            y = [self._aug_fn(i, mag) for i in y]
        return x, y
