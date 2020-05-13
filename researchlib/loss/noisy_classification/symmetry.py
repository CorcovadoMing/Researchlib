from torch import nn
import torch
import torch.nn.functional as F


class SymmetryNLLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1)).to(x.dtype)
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
        ce = -(y * x.log()).sum(-1)
        rce = -(x * y.log()).sum(-1)
        loss = self.alpha * ce + self.beta * rce
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss