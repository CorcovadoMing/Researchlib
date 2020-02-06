import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class TranslateX(namedtuple('TranslateX', ())):
    def __call__(self, x, choice, v):
        if choice:
            x = _to_pil(x)
            x = x.transform(x.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
            x = _to_numpy(x)
        return x

    def options(self):
        return [{
            'choice': b,
            'v': v
        } for b in [True, False] for v in np.linspace(-0.45, 0.45)]

