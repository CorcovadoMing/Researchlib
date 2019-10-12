import torch
import numpy as np
from functools import singledispatch
from collections import namedtuple

@singledispatch
def _pad(x, w):
    raise NotImplementedError

@_pad.register(torch.Tensor)
def _(x, w):
    return torch.nn.ReflectionPad2d(w)(x[None, :, :, :])[0]

@_pad.register(np.ndarray)
def _(x, w):
    return np.pad(x, pad_width=((0, 0), (w, w), (w, w)), mode='reflect')


# Interface
class Crop(namedtuple('Crop', ('h', 'w', 'pad'))):
    def __call__(self, x, choice, x0, y0):
        if choice:
            x_pad = _pad(x, self.pad)
            x = x_pad[..., y0:y0+self.h, x0:x0+self.w]
        return x

    def options(self):
        W, H = self.w + self.pad, self.h + self.pad
        return [{'choice': b, 'x0': x0, 'y0': y0} for b in [True, False] for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]