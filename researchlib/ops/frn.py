import torch
from torch import nn


class _FRN2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps) + 1e-6)
        return self.gamma * x + self.beta