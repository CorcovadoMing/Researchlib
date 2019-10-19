import torch
from collections import namedtuple


class Categorical(namedtuple('Categorical', [])):
    def __call__(self, x, y):
        _, pred = torch.max(x.detach(), 1)
        return pred, y.detach()
