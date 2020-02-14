import torch
from collections import namedtuple


class BinaryCategorical(namedtuple('BinaryCategorical', [])):
    def __call__(self, x):
        x = x.detach()
        pred = (x > 0.5)
        return pred
