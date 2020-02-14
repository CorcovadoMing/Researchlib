import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class Rotate(namedtuple('Rotate', ())):
    def __call__(self, x, choice, v):
        if choice:
            x = _to_pil(x)
            x = x.rotate(v)
            x = _to_numpy(x)
        return x

    def options(self, prob=0.5):
        return [{
            'choice': b,
            'v': v
        } for b in np.random.choice([True, False], p=[prob, 1-prob], size=1) for v in np.linspace(-30, 30)]

