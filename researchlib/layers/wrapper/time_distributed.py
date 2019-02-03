from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x, time_dim=1):
        b, t = x.size(0), x.size(1)
        x = x.contiguous().view(-1, *x.shape[2:])
        x = self.f(x)
        return x.view(b, t, *x.shape[1:])