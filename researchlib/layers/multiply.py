from torch import nn


class _Multiply(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        return x * self.ratio
