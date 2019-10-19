from torch import nn
import torch


class _SupportFeatureConcat(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        x2 = x2.expand(x1.size(0), *[-1 for _ in range(x2.dim())])
        x1 = x1.unsqueeze(1).expand_as(x2)
        x2 = torch.cat([x1, x2], dim=2)
        return x2