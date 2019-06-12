import torch
from torch import nn

class MultiscaleOutput(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, x):
        return self.f(x)
        
        
class MultiscaleInput(nn.Module):
    def __init__(self, f, type='concat'):
        super().__init__()
        self.f = f
        self.type = type
        
    def forward(self, x, x_p):
        if self.type == 'concat':
            x = torch.cat([x, x_p], dim=1)
        elif self.type == 'add':
            x = x + x_p
        return self.f(x)