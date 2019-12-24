from torch import nn
import torch


class _ActiveNoise(nn.Module):
    def __init__(self, channel, size, mix_type='mixed', learn_type=None):
        super().__init__()
        if learn_type == 'pixel':
            self.alpha = nn.Parameter(torch.ones(1, channel, size, size))
            self.beta = nn.Parameter(torch.ones(1, channel, size, size))
        elif learn_type == 'channel':
            self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
            self.beta = nn.Parameter(torch.ones(1, channel, 1, 1))
        elif learn_type == 'sample':
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.beta = nn.Parameter(torch.ones(1, 1, 1, 1))
        else:
            self.alpha = 1
            self.beta = 1
        self.mix_type = mix_type
        self.learn_type = learn_type
    
    def forward(self, x):
        x = x - x.mean([-1, -2], keepdim=True)
        x = x / x.std([-1, -2], keepdim=True)
        if self.training:
            if self.mix_type == 'mul' or self.mix_type == 'mixed':
                mul_noise = torch.empty_like(x).to(x.device).normal_(0, 1)
                mul_noise = torch.tanh(mul_noise) + 1
                x = x * (mul_noise * self.alpha.expand_as(x))
            if self.mix_type == 'add' or self.mix_type == 'mixed':
                add_noise = torch.empty_like(x).to(x.device).normal_(0, 1)
                add_noise = torch.tanh(add_noise)
                x = x + (add_noise * self.beta.expand_as(x))
        else:
            if self.mix_type == 'mul' or self.mix_type == 'mixed':
                x = x * self.alpha.expand_as(x)
        return x