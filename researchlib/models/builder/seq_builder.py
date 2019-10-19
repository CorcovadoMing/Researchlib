from torch import nn
import torch


class _Seq(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(*model) if type(model) == list else model for model in models
        ])

    def forward(self, *x):
        for model in self.models:
            if type(x) != torch.Tensor:
                x = model(*x)
            else:
                x = model(x)
        return x
