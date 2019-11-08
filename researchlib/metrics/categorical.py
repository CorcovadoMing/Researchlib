import torch
from collections import namedtuple


class Categorical(namedtuple('Categorical', [])):
    def __call__(self, x):
        x = x.detach()
        _, pred = torch.max(x, 1)
        return pred
