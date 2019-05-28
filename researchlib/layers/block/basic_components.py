import torch
from torch import nn
import torch.nn.functional as F
from ...models import builder

class _DownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
        self.a = nn.AvgPool2d(pooling_factor)
        self.c = builder([
                    nn.Conv2d(in_dim, in_dim, 3, pooling_factor, 1),
                    nn.LeakyReLU(0.2)
                    ])
        self.red = builder([
                    nn.Conv2d(in_dim*3, in_dim, 1),
                    nn.LeakyReLU(0.2)
                    ])
    
    def forward(self, x):
        x = torch.cat([self.m(x), self.a(x), self.c(x)], dim=1)
        return self.red(x)
     

class _UpSampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.activator = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.activator(x)