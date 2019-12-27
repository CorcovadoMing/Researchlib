from torch import nn
import torch
import torch.nn.functional as F

class _ActiveNoise(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            add_noise = torch.empty_like(x).to(x.device).normal_(0, 2*float(x.std()))
            add_noise -= add_noise.mean(axis=list(range(2, add_noise.ndim)), keepdim=True).expand_as(add_noise) # mean calibration
            x = x + add_noise
        return x