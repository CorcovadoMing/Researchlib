import torch
from torch import nn
from .adaptive_concat_pool import *

class PairLayer1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, kernel_size, padding)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activator = nn.SELU(inplace=True)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        return self.dp(self.activator(self.bn(self.conv(x))))
        
        
class PairSELayer1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, l=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, kernel_size, padding)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activator = nn.SELU(inplace=True)
        self.dp = nn.Dropout(0.5)
        
        self.se = AdaptiveConcatPool1d(l)
        self.se_f = nn.Conv1d(int(out_dim*2), int(out_dim/2), 1)
        self.se_a = nn.ReLU(inplace=True)
        self.se_r = nn.Conv1d(int(out_dim/2), out_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        f = self.se(x)
        f = self.se_f(f)
        f = self.se_a(f)
        f = self.se_r(f)
        f = torch.sigmoid(f)
        x = x * f
        x = self.bn(x)
        x = self.activator(x)
        x = self.dp(x)
        return x