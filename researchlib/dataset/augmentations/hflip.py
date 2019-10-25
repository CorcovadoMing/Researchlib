import torch
import numpy as np
from functools import singledispatch
from collections import namedtuple


@singledispatch
def _hflip(x):
    raise NotImplementedError


@_hflip.register(torch.Tensor)
def _(x):
    return torch.flip(x, [-1])


@_hflip.register(np.ndarray)
def _(x):
    return x[..., ::-1].copy()


# Interface
class HFlip(namedtuple('HFlip', ())):
    def __call__(self, x, choice):
        return _hflip(x) if choice else x

    def options(self):
        return [{'choice': b} for b in [True, False]]
