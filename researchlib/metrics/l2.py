from collections import namedtuple
import torch


class L2(namedtuple('L2', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach().view_as(x)
        return (x - y).pow(2).mean()
