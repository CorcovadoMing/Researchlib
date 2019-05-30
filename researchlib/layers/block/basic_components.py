import torch
from torch import nn
import torch.nn.functional as F

class _DownSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.m = nn.MaxPool2d(pooling_factor)
        self.a = nn.AvgPool2d(pooling_factor)
        self.c = nn.Conv2d(in_dim, in_dim, 3, pooling_factor, 1)
        self.red = nn.Conv2d(in_dim*3, in_dim, 1)
        self.activator = nn.LeakyReLU(0.2)
        self.preact = preact
    
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = torch.cat([self.m(x), self.a(x), self.c(x)], dim=1)
        x = self.activator(x)
        x = self.red(x)
        if not self.preact: x = self.activator(x)
        return x

class _UpSampling(nn.Module):
    def __init__(self, in_dim, pooling_factor, preact=False):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.activator = nn.LeakyReLU(0.2)
        self.preact = preact
        self.pooling_factor = pooling_factor
        
    def forward(self, x):
        if self.preact: x = self.activator(x)
        x = F.interpolate(x, scale_factor=self.pooling_factor)
        x = self.conv(x)
        if not self.preact: x = self.activator(x)
        return x
        