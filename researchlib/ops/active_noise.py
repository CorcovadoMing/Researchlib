from torch import nn
import torch


class _ActiveNoise(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            add_noise = torch.empty_like(x).to(x.device).normal_(0, 2*float(x.std()))
            x = x + add_noise
        return x