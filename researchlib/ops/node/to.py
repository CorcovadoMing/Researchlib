from torch import nn


class _To(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
    
    def forward(self, x):
        return x.to(self.arg)