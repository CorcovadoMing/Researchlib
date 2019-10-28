from torch import nn
import torch


class _ActiveNoise(nn.Module):
    def __init__(self, channel, type='mixed', learnable=False):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
            self.beta = nn.Parameter(torch.ones(1, channel, 1, 1))
        else:
            self.alpha = 1
            self.beta = 1
        self.type = type
    
    def forward(self, x):
        if self.training:
            noise = torch.empty_like(x).to(x.device).uniform_()
            if self.type == 'mul' or self.type == 'mixed':
                return x * noise * self.alpha.expand_as(x)
            if self.type == 'add' or self.type == 'mixed':
                return x + (noise - 0.5) * self.beta.expand_as(x)
        return x