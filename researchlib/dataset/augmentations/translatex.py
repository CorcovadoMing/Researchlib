import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class TranslateX(namedtuple('TranslateX', ())):
    def __call__(self, x, y, choice, v):
        if choice:
            x = _to_pil(x)
            x = x.transform(x.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
            x = _to_numpy(x)
        return x, y

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'v': np.random.choice(np.linspace(-0.5, 0.5))
        }
