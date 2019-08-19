from torch import nn


class _Squeeze(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim:
            return x.squeeze(self.dim)
        else:
            return x.squeeze()
