from torch import nn
import torch
import torch.nn.functional as F


class JointOptimizationLoss(nn.Module):
    def __init__(self, alpha=1.2, beta=0.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1)).to(x.dtype)
        x_avg = x.mean(0)
        prior = torch.ones(x.size(1)).to(x.dtype).to(x.device) / x.size(1)
        x_avg = torch.clamp(x_avg, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-6, max=1.0)
        x = torch.clamp(x, min=1e-6, max=1.0)
        loss_p = -(prior * x_avg.log()).sum(-1).mean()
        loss_e = -(y * y.log()).sum(-1).mean()
        loss_n = -(y * x.log()).sum(-1).mean()
        return loss_n + self.alpha * loss_p + self.beta * loss_e