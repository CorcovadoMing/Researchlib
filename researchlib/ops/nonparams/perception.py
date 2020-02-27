from torch import nn
import torch
from .sobel import _SobelHorizontal2d, _SobelVertical2d

class _Perception(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.sobel_x = _SobelHorizontal2d(in_dim)
        self.sobel_y = _SobelVertical2d(in_dim)
    
    def forward(self, x):
        return torch.cat([
            self.sobel_x(x),
            self.sobel_y(x),
            x
        ], 1)