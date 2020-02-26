from torch import nn
import torch
import torch.nn.functional as F


class _SobelHorizontal2d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.prior = torch.tensor([[[
                        [-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.],
                    ]]]).repeat(self.in_dim, 1, 1, 1)
    
    def forward(self, x):
        return F.conv2d(x, self.prior, padding=1, groups=self.in_dim)


class _SobelVertical2d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.prior = torch.tensor([[[
                        [-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.],
                    ]]]).repeat(self.in_dim, 1, 1, 1)
    
    def forward(self, x):
        return F.conv2d(x, self.prior, padding=1, groups=self.in_dim)