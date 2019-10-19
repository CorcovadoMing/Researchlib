import torch
import numpy as np
from functools import singledispatch


@singledispatch
def format_transpose(x, source, target):
    raise NotImplementedError


@format_transpose.register(torch.Tensor)
def _(x, source, target):
    return x.permute([source.index(d) for d in target])


@format_transpose.register(np.ndarray)
def _(x, source, target):
    return x.transpose([source.index(d) for d in target])
