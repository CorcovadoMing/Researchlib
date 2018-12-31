from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.f = f

    def forward(self, x): 
        print(x.shape)
        out = []
        for i in range(x.shape[1]):
            out.append(self.f(x[:, i, :, :, :]))
        out = torch.stack(out, dim=1)
        return out