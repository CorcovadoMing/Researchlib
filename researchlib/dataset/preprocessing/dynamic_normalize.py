import torch
import numpy as np
from functools import singledispatch
from .normalize import normalize


@singledispatch
def _collect(x):
    raise NotImplementedError


@_collect.register(torch.Tensor)
def _(x):
    return x.mean(dim = tuple(range(x.dim() - 1))), x.std(dim = tuple(range(x.dim() - 1)))


@_collect.register(np.ndarray)
def _(x):
    return x.mean(axis = tuple(range(x.ndim - 1))), x.std(axis = tuple(range(x.ndim - 1)))


class DynamicNormalize:
    def __init__(self):
        self.mean = np.array([0, 0, 0])
        self.std = np.array([1, 1, 1])

    def __call__(self, x):
        mean, std = map(np.array, _collect(x))
        self.mean = 0.9 * self.mean + 0.1 * mean
        self.std = 0.9 * self.std + 0.1 * std
        return normalize(x, self.mean, self.std)
