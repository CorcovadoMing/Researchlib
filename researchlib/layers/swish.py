import torch
from torch import nn

class _Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)
