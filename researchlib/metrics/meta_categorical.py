import torch
from collections import namedtuple


class MetaCategorical(namedtuple('MetaCategorical', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        _, y = torch.max(y, -1)
        y, _ = torch.max(y, -1)
        x = x.sum(1)
        _, x = torch.max(x, -1)
        return x, y