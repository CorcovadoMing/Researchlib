import numpy as np
from .utils import _to_pil, _to_numpy
from collections import namedtuple
import PIL


class Solarize(namedtuple('Solarize', ())):
    def __call__(self, x, choice, v):
        if choice:
            x = _to_pil(x)
            x = PIL.ImageOps.solarize(x, v)
            x = _to_numpy(x)
        return x

    def options(self):
        return [{
            'choice': b,
            'v': v
        } for b in [True, False] for v in np.linspace(0, 256)]

