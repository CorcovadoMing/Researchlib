from ..template import template
from ...utils import mapping
import numpy as np
import torch


def _grayscale(img):
    gs = img.clone()
    gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
    gs[1].copy_(gs[0])
    gs[2].copy_(gs[0])
    return gs


class Grayscale(template.TorchAugmentation):

    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag

    def _aug_fn(self, img):
        return _grayscale(img)

    def forward_single(self, x, y, mag):
        x = [self._aug_fn(i) for i in x]
        if self.include_y:
            y = [self._aug_fn(i) for i in y]
        return x, y
