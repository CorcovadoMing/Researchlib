from torch import nn


class _NoOp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x