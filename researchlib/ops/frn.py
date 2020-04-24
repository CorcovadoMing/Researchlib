import torch
from torch import nn


class _FRN2d(nn.Module):
    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = nn.Parameter(torch.Tensor([eps]))
    
    def forward(self, x):
        nu2 = torch.square(x).mean(dim=[-1, -2], keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps) + 1e-3)
        return self.gamma * x + self.beta