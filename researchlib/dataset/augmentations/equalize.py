import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class Equalize(namedtuple('Equalize', ())):
    def __call__(self, x, choice):
        if choice:
            x = _to_pil(x)
            x = PIL.ImageOps.equalize(x)
            x = _to_numpy(x)
        return x

    def options(self, prob=0.5):
        return [{'choice': b} for b in np.random.choice([True, False], p=[prob, 1-prob], size=1)]