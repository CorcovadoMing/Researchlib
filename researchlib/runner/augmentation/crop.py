from ..template import template
from ...utils import mapping
import torch


class Crop2d(template.TorchAugmentation):
    def __init__(self, prob = None, mag = None, include_y = False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag

    def _aug_fn(self, img, padding):
        if padding == 0:
            return img
        else:
            new_img = torch.zeros(
                img.size(0),
                img.size(1) + 2 * padding,
                img.size(2) + 2 * padding
            )
            new_img[:, padding:-padding, padding:-padding] = img
            random = torch.randint(low = 0, high = 2 * padding, size = (1, 2)).squeeze()
            return new_img[:, random[0]:random[0] + img.size(1), random[1]:random[1] + img.size(2)]

    def forward_single(self, x, y, mag):
        s = x[0].size(-1)
        padding = mapping(mag, [0, 1], [0, s // 4], to_int = True)
        x = [self._aug_fn(i, padding) for i in x]
        if self.include_y:
            y = [self._aug_fn(i, padding) for i in y]
        return x, y
