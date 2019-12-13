import torch
from torch import nn
from box_convolution import BoxConv2d
import numpy as np


class _BoxConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, dilation=1, bias=False, groups=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.box_nums = 8
        
        if kernel_size != 1 and stride == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim // self.box_nums, 1, bias=False),
                nn.BatchNorm2d(self.out_dim // self.box_nums),
                BoxConv2d(self.out_dim // self.box_nums, self.box_nums, 1, 1, 0.5),
            )
            self.conv2 = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
            
            self.initialized = False
            self.box = True
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
            )
            self.initialized = True
            self.box = False
    
    
    def reset_parameters():
        self.initialized = False
        
    
    def forward(self, x):
        if not self.initialized:
            h, w = x.shape[-2:]
            self.conv1[-1].max_input_h, self.conv1[-1].max_input_w = h, w
            self.conv1[-1].reset_parameters()
            self.initialized = True
        if self.box:
            return self.conv1(x) + self.conv2(x)
        else:
            return self.conv(x)
    
    
    def draw_boxes(self):
        if self.box:
            return self.conv1[-1].draw_boxes(resolution=(600, 600), weights=torch.ones(self.in_dim, self.box_nums))
        else:
            return np.zeros((600, 600, 3))
        