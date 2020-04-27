import torch
from torch import nn


class _TLU2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        cached_type = x.dtype
        x = torch.max(x.float(), self.tau)
        return x.to(cached_type)