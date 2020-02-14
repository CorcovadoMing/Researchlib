import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class TranslateY(namedtuple('TranslateY', ())):
    def __call__(self, x, choice, v):
        if choice:
            x = _to_pil(x)
            x = x.transform(x.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
            x = _to_numpy(x)
        return x

    def options(self, prob=0.5):
        return [{
            'choice': b,
            'v': v
        } for b in np.random.choice([True, False], p=[prob, 1-prob], size=1) for v in np.linspace(-0.45, 0.45)]


