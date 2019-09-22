from ..template import template
from ...utils import mapping
import torch


class SamplePairing(template.TorchAugmentation):
    def __init__(self, prob = None, mag = None, include_y = False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag

    def _aug_fn(self, img, mag):
        index = torch.randperm(img.size(0))
        ratio = mag / 2
        return (1 - ratio) * img + ratio * img[index]

    def forward_batch(self, x, y, mag):
        x = [self._aug_fn(i, mag) for i in x]
        if self.include_y:
            y = [self._aug_fn(i, mag) for i in y]
        return x, y
