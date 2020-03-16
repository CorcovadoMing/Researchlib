import torch
from torch import nn
import torch.nn.functional as F


class _Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.temperature.requires_grad = False
        
    def forward(self, x):
        if self.training:
            return F.softmax(x, self.dim)
        else:
            return F.softmax(x/self.temperature, self.dim)
        
        

class _LogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.temperature.requires_grad = False
        
    def forward(self, x):
        if self.training:
            return F.log_softmax(x, self.dim)
        else:
            return F.log_softmax(x/self.temperature, self.dim)
        
        
class _GumbelSoftmax(nn.Module):
    def __init__(self, dim=-1, tau=1, hard=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim
    
    def forward(self, x):
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=self.dim)