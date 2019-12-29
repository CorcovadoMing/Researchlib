from torch import nn
import torch


class _ActiveNoise(nn.Module):
    def __init__(self, noise_type):
        super().__init__()
        self.noise_type = noise_type
        
    def forward(self, x):
        if self.training:
            std = float(x.std())
            if 'mul' in self.noise_type:
                mul_noise = torch.empty_like(x).to(x.device).normal_(0, 0.01)
                x = x + x * mul_noise
            if 'add' in self.noise_type:
                add_noise = torch.empty_like(x).to(x.device).normal_(0, 2 * std)
                x = x + add_noise
        return x