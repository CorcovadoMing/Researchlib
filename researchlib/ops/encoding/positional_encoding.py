from torch import nn
import torch
import math


class _PositionalEncoding1d(nn.Module):
    def __init__(self, features, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2).float() * (-math.log(10000.0) / features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1), :]
        return x