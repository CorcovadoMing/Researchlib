import torch
import torch.nn as nn
from torch.nn import init


class _TemplateBank(nn.Module):
    def __init__(self, num_templates, in_planes, out_planes, kernel_size):
        super().__init__()
        self.coefficient_shape = (num_templates,1,1,1,1)
        templates = [torch.Tensor(out_planes, in_planes, kernel_size, kernel_size) for _ in range(num_templates)]
        for i in range(num_templates): init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(torch.stack(templates))
        
    def forward(self, coefficients):
        return (self.templates*coefficients).sum(0)