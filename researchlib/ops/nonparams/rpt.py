import torch
from torch import nn


class _RPT(nn.Module):
    def __init__(self, r=1):
        super().__init__()
        self.r = r
    
    def forward(self, x):
        return x + self.r * torch.empty_like(x).normal_()