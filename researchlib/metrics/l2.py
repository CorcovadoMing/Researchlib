from collections import namedtuple
import torch


class L2(namedtuple('L2', [])):
    def __call__(self, x, y):
        return (x.detach() - y.detach()).pow(2).sum().float() / x.size(0)
