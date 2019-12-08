from torch import nn
import torch


class _CombinePool2d(nn.Module):
    def __init__(self, kernel_size, out_dim):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size)
        self.conv_pool = nn.Conv2d(out_dim, out_dim, 4, kernel_size, 1)
        self.reduce = nn.Conv2d(out_dim*3, out_dim, 1)
    
    def forward(self, x):
        return self.reduce(torch.cat([self.avg_pool(x), self.max_pool(x), self.conv_pool(x)], dim=1))
        