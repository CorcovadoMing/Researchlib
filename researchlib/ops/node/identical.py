from torch import nn


class _Identical(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x
