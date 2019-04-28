import torch
from torch import nn

class _AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
    
    
class _AdaptiveConcatPool1d(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.ap = nn.AvgPool1d(l)
        self.mp = nn.MaxPool1d(l)
    
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)