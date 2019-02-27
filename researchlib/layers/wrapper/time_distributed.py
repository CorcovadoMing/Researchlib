from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x, time_dim=1):
        x = x.transpose(1, 2)
        b, t = x.size(0), x.size(1)
        x = x.contiguous().view(-1, *x.shape[2:])
        x = self.f(x)
        x = x.view(b, t, *x.shape[1:])
        x = x.transpose(1, 2)
        return x