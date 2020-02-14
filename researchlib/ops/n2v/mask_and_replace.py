from torch import nn
import torch


class _MaskAndReplace(nn.Module):
    def __init__(self, num=1):
        super().__init__()
        self.num = num
    
    def forward(self, x):
        if self.training:
            mask = torch.ones_like(x)
            pool_x = torch.randperm(x.size(-2))
            pool_y = torch.randperm(x.size(-1))
            mask[:, :, pool_x[:self.num], pool_y[:self.num]] = 0
            x = x * mask
            x[:, :, pool_x[:self.num], pool_y[:self.num]] = x[:, :, pool_x[-self.num:], pool_y[-self.num:]]
            loc = (pool_x[:self.num], pool_y[:self.num])
        else:
            loc = (0, 0)
        return x, loc