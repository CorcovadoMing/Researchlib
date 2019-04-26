from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x):
        index = list(range(x.dim()))
        index = [index[0], index[-1]] + index[1:-1]
        x = x.permute(*index)
        bs, ts = x.size(0), x.size(1)
        x = self.f(x.contiguous().view(bs*ts, *x.shape[2:]))
        x = x.view(bs, ts, *x.shape[1:])
        index = list(range(x.dim()))
        index = [index[0]] + index[2:] + [index[1]]
        x = x.permute(*index)
        return x