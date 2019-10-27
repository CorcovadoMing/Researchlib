from torch import nn
import torch.nn.functional as F


class _Resize(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        return F.interpolate(x, self.size)