from torch import nn


class _Equal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x == y