from torch import nn


class _Sample(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.sample().item()