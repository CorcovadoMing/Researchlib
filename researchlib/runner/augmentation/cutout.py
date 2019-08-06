from ..template import template
from ...utils import mapping
import numpy as np
import torch

class Cutout(template.TorchAugmentation):
    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag
    
    def _aug_fn(self, img, length):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(1):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
    
    def forward_single(self, x, y, mag):
        max_length = x[0].size(1) // 2
        length = mapping(mag, [0, 1], [0, max_length], to_int=True)
        x = [ self._aug_fn(i, length) for i in x]
        if self.include_y:
            y = [ self._aug_fn(i, length) for i in y]
        return x, y