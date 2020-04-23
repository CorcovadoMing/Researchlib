from torch import nn
import torch
import torch.nn.functional as F


class _GhostConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, **kwargs):
        super().__init__()
        
        self.normal_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, **kwargs)
        self.cheap_conv = nn.Conv2d(out_channels//2, out_channels//2, kernel_size, groups=out_channels//2, stride=1, padding=padding, dilation=dilation, bias=bias) 
    
    def forward(self, x):
        x = self.normal_conv(x)
        x_ = self.cheap_conv(x)
        return torch.cat([x, x_], dim=1)