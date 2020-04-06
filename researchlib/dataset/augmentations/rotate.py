import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class Rotate(namedtuple('Rotate', ())):
    def __call__(self, x, choice, v, c):
        if choice:
            x = _to_pil(x)
            x = x.rotate(v, fillcolor=c)
            x = _to_numpy(x)
        return x

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'v': np.random.choice(np.linspace(-45, 45)),
            'c': np.random.randint(0, 255)
        }