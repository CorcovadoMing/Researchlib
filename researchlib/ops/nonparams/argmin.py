from torch import nn


class _Argmin(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.argmax(-1)