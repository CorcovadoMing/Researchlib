from torch import nn
import torch
import torch.nn.functional as F


class IMAENLLoss(nn.Module):
    def __init__(self, T=4, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.T = T
        
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1)).to(x.dtype)
        
        mae = F.l1_loss(x, y, reduction='none')
        
        with torch.no_grad():
            x_detach = x.clone().detach()
            weighting_factor = (x_detach.clamp_(1e-4, 1) * (1 - x_detach).clamp_(1e-4, 1)).clamp_(1e-4, 1)
            weighting = (self.T * weighting_factor).exp()
        
        loss = (mae * weighting).sum(-1)
        
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss