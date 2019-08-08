from ..template import template
from ...utils import mapping
from .pil_helper import _PILHelper

class AutoContrast(template.NumpyAugmentation):
    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag
        self.helper = _PILHelper()
    
    def _aug_fn(self, img, mag):
        img = self.helper.to_pil(img)
        cutoff = int(mag * 49)
        img = ImageOps.autocontrast(img, cutoff=cutoff)
        return self.helper.to_numpy(img)
    
    def forward_single(self, x, y, mag):
        x = [ self._aug_fn(i, mag) for i in x]
        if self.include_y:
            y = [ self._aug_fn(i, mag) for i in y]
        return x, y