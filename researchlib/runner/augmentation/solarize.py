from ..template import template
from ...utils import mapping
from .pil_helper import _PILHelper
from PIL import Image, ImageOps, ImageEnhance


class Solarize(template.NumpyAugmentation):
    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag
        self.helper = _PILHelper()

    def _aug_fn(self, img, mag):
        img = self.helper.to_pil(img.transpose(1, 2, 0))
        threshold = (1 - mag) * 255
        img = ImageOps.solarize(img, threshold)
        img = self.helper.to_numpy(img).transpose(2, 0, 1)
        return img

    def forward_single(self, x, y, mag):
        x = [self._aug_fn(i, mag) for i in x]
        if self.include_y:
            y = [self._aug_fn(i, mag) for i in y]
        return x, y
