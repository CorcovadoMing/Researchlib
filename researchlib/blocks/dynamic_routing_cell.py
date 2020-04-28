from torch import nn
import torch
import torch.nn.functional as F
from ..ops import op
from .unit.utils import get_dim


class _DynamicRoutingCell(nn.Module):
    def __init__(self, _op, channels, cell_operations = [op.SepConv2d, op.Identical]):
        super().__init__()
        
        dim = get_dim(_op)
        
        self.cell_operations = nn.ModuleList([
            i(channels, channels, kernel_size=3, padding=1, stride=1) for i in cell_operations
        ])
        
        self.in_transform = nn.ModuleList([
            nn.Sequential(
                nn.__dict__[f'Conv{dim}'](channels*2, channels, 1),
                nn.__dict__[f'UpsamplingBilinear{dim}'](scale_factor=2)
            ),
            op.Identical(),
            nn.__dict__[f'Conv{dim}'](channels//2, channels, 1, 2, 0),
        ])
        
        self.out_transform = nn.ModuleList([
            nn.__dict__[f'Conv{dim}'](channels, channels*2, 1, 2, 0),
            op.Identical(),
            nn.Sequential(
                nn.__dict__[f'Conv{dim}'](channels, channels//2, 1),
                nn.__dict__[f'UpsamplingBilinear{dim}'](scale_factor=2)
            )
        ])
        
        self.soft_gate = nn.Sequential(
            nn.__dict__[f'Conv{dim}'](channels, channels, 3, 1, 1),
            nn.__dict__[f'BatchNorm{dim}'](channels),
            nn.ReLU(inplace=True),
            nn.__dict__[f'AdaptiveAvgPool{dim}'](1),
            nn.__dict__[f'Conv{dim}'](channels, 3, 1),
            nn.Tanh(),
            nn.ReLU(inplace=True)
        )
        
        self.post_fix = [
            lambda x, y, check: y,
            lambda x, y, check: x * check + y * (1-check),
            lambda x, y, check: y,
        ]
    
    def forward(self, *inputs):
        aggregated_x = sum([j(i) for i, j in zip(inputs, self.in_transform)])
        out = sum([i(aggregated_x) for i in self.cell_operations])
        gates = self.soft_gate(out)
        mask = (gates.sum(1, keepdim=True) == 0).to(gates.dtype)
        out = aggregated_x * mask + out * (1-mask)
        result = [i(out) * g for i, g, in zip(self.out_transform, gates.split(1, 1))]
        result = [fix(out, y, mask) for fix, y in zip(self.post_fix, result)]
        return result