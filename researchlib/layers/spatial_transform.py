import torch
from torch import nn
import torch.nn.functional as F
from . import *

class _SpatialTransform(nn.Module):
    def __init__(self, localization, out='transform'):
        super().__init__()
        self.localization = localization
        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.out = out

    def transform(self, x):
        xs = self.localization(x)        
        xs = xs.view(-1, 2, 3)
        grid = F.affine_grid(xs, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        if self.out == 'transform':
            x = self.transform(x)
        elif self.out == 'concat':
            x = torch.cat([x, self.transform(x)], dim=1)
        else:
            print(self.out, ' is not defined.')
        return x