from torch import nn
import torch
import math


class _AEDecoder2d(nn.Module):
    def __init__(self, blocks, dimension, to_rgb=True, to_grayscale=False):
        super().__init__()
        model = []
        cur_dim = dimension
        for _ in range(blocks):
            model += [
                nn.ConvTranspose2d(cur_dim, cur_dim//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cur_dim//2),
                nn.ReLU(inplace=True)
            ]
            cur_dim = cur_dim//2
        assert not (to_rgb and to_grayscale)
        if to_rgb:
            model += [
                nn.Conv2d(cur_dim, 3, 1),
            ]
        if to_grayscale:
            model += [
                nn.Conv2d(cur_dim, 1, 1),
            ]
        self.f = nn.Sequential(*model)
    
    def forward(self, x):
        return self.f(x)
    

class _VAEDecoder2d(nn.Module):
    def __init__(self, target_size, dimension, to_rgb=True, to_grayscale=False):
        super().__init__()
        from . import op
    
        model = []
        
        self.mean_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dimension, dimension),
        )
        
        self.logvar_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dimension, dimension),
        )
        
        
        self.to2d = nn.Sequential(
            nn.Linear(dimension, dimension * 4, bias=False),
            op.Reshape(-1, dimension, 2, 2),
            nn.BatchNorm2d(dimension),
            nn.ReLU()
        )
        
        blocks = target_size
        cur_dim = dimension
        for _ in range(int(math.log2(target_size) - 1)):
            model += [
                nn.ConvTranspose2d(cur_dim, cur_dim//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cur_dim//2),
                nn.ReLU(inplace=True)
            ]
            cur_dim = cur_dim//2
        assert not (to_rgb and to_grayscale)
        if to_rgb:
            model += [
                nn.Conv2d(cur_dim, 3, 1),
            ]
        if to_grayscale:
            model += [
                nn.Conv2d(cur_dim, 1, 1),
            ]
        self.decode = nn.Sequential(*model)
        self.dimension = dimension
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.training:
            mu, logvar = self.mean_branch(x), self.logvar_branch(x)
            z = self.reparameterize(mu, logvar)
            z = self.to2d(z)
        else:
            mu, logvar = torch.zeros(x.size(0), self.dimension, device=x.device, dtype=x.dtype), torch.zeros(x.size(0), self.dimension, device=x.device, dtype=x.dtype)
            z = self.reparameterize(mu, logvar)
            z = self.to2d(z)
        return self.decode(z), mu, logvar