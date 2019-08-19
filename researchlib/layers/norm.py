from torch import nn
import torch


class _Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.norm(x, dim=-1)
