import torch
from torch import nn


class _TLU2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        return torch.max(x, self.tau)