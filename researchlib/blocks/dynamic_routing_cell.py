from torch import nn
import torch
import torch.nn.functional as F
from ..ops import op
from .unit.utils import get_dim
from functools import partial


class _DynamicRoutingCell(nn.Module):
    def __init__(self, _op, channels, cell_operations = [nn.Conv2d, op.Identical], allow_up=True, allow_down=True):
        super().__init__()
        
        dim = get_dim(_op)
        
        self.allow_up = allow_up
        self.allow_down = allow_down
        
        self.cell_operations = nn.ModuleList([
            nn.Sequential(
                i(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels//8, bias=False),
                nn.__dict__[f'BatchNorm{dim}'](channels),
            ) if i != op.Identical else i() for i in cell_operations
        ])
        
        self.out_transform = nn.ModuleList([
            nn.Sequential(
                op.__dict__[f'Conv{dim}'](channels, channels//2, 1, bias=False),
                nn.__dict__[f'BatchNorm{dim}'](channels//2),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2.0),
            ),
            op.Identical(),
            nn.Sequential(
                op.__dict__[f'Conv{dim}'](channels, channels*2, 1, 2, bias=False),
                nn.__dict__[f'BatchNorm{dim}'](channels*2),
                nn.ReLU(inplace=True),
            ),
        ])
        
        self.soft_gate = nn.Sequential(
            nn.__dict__[f'Conv{dim}'](channels, channels, 3, 1, 1, bias=False),
            nn.__dict__[f'BatchNorm{dim}'](channels),
            nn.ReLU(inplace=True),
            nn.__dict__[f'AdaptiveAvgPool{dim}'](1),
            nn.__dict__[f'Conv{dim}'](channels, 3, 1),
            nn.Tanh(),
            nn.ReLU()
        )
        
        self.post_fix = [
            lambda x, y, check: y,
            lambda x, y, check: x * check + y * (1-check),
            lambda x, y, check: y,
        ]
        
        if not self.allow_up:
            self.out_transform = self.out_transform[1:]
            self.post_fix = self.post_fix[1:]
        
        if not self.allow_down:
            self.out_transform = self.out_transform[:-1]
            self.post_fix = self.post_fix[:-1]
            
    
    def forward(self, inputs):
        aggregated_x = sum(inputs)
        out = F.relu(sum([i(aggregated_x) for i in self.cell_operations]), inplace=True)
        gates = self.soft_gate(aggregated_x)
        mask = (gates.sum(1, keepdim=True) == 0).to(gates.dtype)
        out = aggregated_x * mask + out * (1-mask)
        result = [i(out) * g for i, g, in zip(self.out_transform, gates.split(1, 1))]
        result = [fix(out, y, mask) for fix, y in zip(self.post_fix, result)]
        
        if not self.allow_up:
            result = [None] + result
        if not self.allow_down:
            result = result + [None]
            
        return result