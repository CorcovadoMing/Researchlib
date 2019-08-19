from torch import nn


class _Permute(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)
