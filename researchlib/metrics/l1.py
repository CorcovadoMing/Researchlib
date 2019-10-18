from collections import namedtuple
import torch

class L1(namedtuple('L1', [])):
    def __call__(self, x, y):
        return torch.abs(x.detach() - y.detach()).sum().float() / x.size(0)