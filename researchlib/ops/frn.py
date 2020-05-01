import torch
from torch import nn


class _FRN2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = nn.Parameter(torch.Tensor([1e-4]))
    
    def forward(self, x):
        nu2 = x.abs().mean(dim=[2, 3], keepdims=True)
        x = x  / torch.max((nu2 + torch.abs(self.eps)), torch.empty_like(nu2).fill_(1e-4))
        return self.gamma * x + self.beta