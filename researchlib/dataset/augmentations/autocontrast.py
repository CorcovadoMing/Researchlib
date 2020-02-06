from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class AutoContrast(namedtuple('AutoContrast', ())):
    def __call__(self, x, choice):
        if choice:
            x = _to_pil(x)
            x = PIL.ImageOps.autocontrast(x)
            x = _to_numpy(x)
        return x

    def options(self):
        return [{'choice': b} for b in [True, False]]