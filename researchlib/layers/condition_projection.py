import torch
from torch import nn

class _ConditionProjection(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, x):
        data, condition = x
        return torch.cat([data, self.f(condition)], dim=1)