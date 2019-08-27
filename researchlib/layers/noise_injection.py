from torch import nn
import torch

class _NoiseInjection(nn.Module):
    def __init__(self, type='add'):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))
        self.type = type
    
    def forward(self, x):
        if self.training:
            noise = torch.empty_like(x).to(x.device).uniform_() # range: [0, 1]
            if self.type == 'add':
                return x + (noise - 0.5) * self.gain
            elif self.type == 'mul':
                return x * noise * self.gain
        else:
            return x