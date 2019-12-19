import torch
from torch import nn


class _Mixture(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        index = torch.randperm(x.size(0))
        x_mix = x[index]
        ratio = (torch.rand(torch.Size([1]), device=x.device, dtype=x.dtype) * 0.9) / 2
        return (1-ratio) * x + ratio * x_mix