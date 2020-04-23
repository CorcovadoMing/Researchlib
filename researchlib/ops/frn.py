import torch
from torch import nn


class _FRN2d(nn.Module):
    def __init__(self, channels, eps=1e-2):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps#nn.Parameter(torch.Tensor([eps]))
    
    def forward(self, x):
        nu2 = torch.square(x).mean(dim=[-1, -2], keepdims=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)