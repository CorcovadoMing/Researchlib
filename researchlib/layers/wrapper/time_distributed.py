from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x):
        bs, feature, rest = x.size(0), x.size(1), x.shape[2:]
        x = x.contiguous().view(bs, *rest, feature)
        x = self.f(x)
        x = x.contiguous().view(bs, -1, *rest)
        return x