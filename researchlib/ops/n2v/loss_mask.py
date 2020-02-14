from torch import nn


class _LossMask(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, loc):
        if self.training:
            return x[:, :, loc[0], loc[1]]
        else:
            return x