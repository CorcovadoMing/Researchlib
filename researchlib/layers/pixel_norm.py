import torch
from torch import nn

class _PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)