import torch
import numpy as np
from functools import singledispatch

@singledispatch
def normalize(x, mean, std, **kwargs):
    raise NotImplementedError

@normalize.register(torch.Tensor)
def _(x, mean, std):
    x -= mean
    x *= 1.0 / std
    return x

@normalize.register(np.ndarray)
def _(x, mean, std):
    x -= mean
    x *= 1.0 / std
    return x