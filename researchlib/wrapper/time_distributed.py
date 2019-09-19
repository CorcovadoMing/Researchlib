from torch import nn
import torch


class _TimeDistributed(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        bs, ts = x.size(0), x.size(1)
        x = self.f(x.view(bs * ts, *x.shape[2:]))
        x = x.view(bs, ts, *x.shape[1:])
        return x
