from torch import nn
import torch


class _ActiveNoise(nn.Module):
    def __init__(self, noise_type, mul_ratio = 1, add_ratio = 1):
        super().__init__()
        self.noise_type = noise_type
        self.mul_ratio = mul_ratio
        self.add_ratio = add_ratio
        
    def forward(self, x):
        if self.training:
            std = float(x.std())
            if 'mul' in self.noise_type:
                x = x + 1
                x = x / 2
                new_std = std / 2
                mul_noise = torch.empty_like(x).to(x.device).normal_(0, 1) * new_std * self.mul_ratio
                x = x + x * mul_noise 
                x = x * 2
                x = x - 1
                
            if 'add' in self.noise_type:
                # additive noise is not pixel-dependent, so re-normalize is not neccessary
                add_noise = torch.empty_like(x).to(x.device).normal_(0, 1) * std * self.add_ratio
                x = x + add_noise
        return x