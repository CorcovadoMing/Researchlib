from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x, time_dim=1): 
        out = []
        for i in range(x.shape[time_dim]):
            out.append(self.f(x[:, i]))
        out = torch.stack(out, dim=time_dim)
        return out