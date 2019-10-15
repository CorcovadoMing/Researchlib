from torch import nn
from ..wrapper import _MultiscaleOutput, _MultiscaleInput
from ..layers import layer
import torch


class MultiApply(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *x):
        return [self.model(i) for i in x]

    
class SupportFeatureConcat(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        x2 = x2.expand(x1.size(0), *[-1 for _ in range(x2.dim())])
        x1 = x1.unsqueeze(1).expand_as(x2)
        x2 = torch.cat([x1, x2], dim=2)
        return x2

    
class Builder(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(*model) if type(model) == list else model for model in models])

    def forward(self, *x):
        for model in self.models:
            if type(x) != torch.Tensor:
                x = model(*x)
            else:
                x = model(x)
        return x