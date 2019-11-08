from collections import namedtuple
import torch


class L1(namedtuple('L1', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        return torch.abs(x - y).sum().float() / x.size(0)
