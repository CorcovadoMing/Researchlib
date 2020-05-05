from torch import nn
import torch.nn.functional as F
import torch



class SmoothNLLoss(nn.Module):
    def __init__(self, smooth = 0.2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1))
        y = y.to(x.dtype)
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
        kl = -x.log().mean(-1)
        xent = -(y * x.log()).sum(-1)
        loss = ((1 - self.smooth) * xent + self.smooth * kl)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss
    

class SmoothNLLLoss(nn.Module):
    def __init__(self, smooth = 0.2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1))
        y = y.to(x.dtype)
        y = torch.clamp(y, min=1e-4, max=1.0)
        kl = -x.mean(-1)
        xent = -(y * x).sum(-1)
        loss = ((1 - self.smooth) * xent + self.smooth * kl)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class NLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1))
        y = y.to(x.dtype)
        y = torch.clamp(y, min=1e-4, max=1.0)
        loss = -(y * x).sum(-1)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss
    
    
class NLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1))
        y = y.to(x.dtype)
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
        loss = -(y * x.log()).sum(-1)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss
    
    
class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, x, y):
        return F.binary_cross_entropy(x, y.float(), reduction=self.reduction)
    
    
class SNLLoss(nn.Module):
    '''
        Symmetry NL
    '''
    def __init__(self, alpha=1, beta=1, reduction='mean'):
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
        