from torch import nn
import torch.nn.functional as F
import torch


class SmoothNLLoss(nn.Module):
    def __init__(self, smooth = 0.2):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, x, y):
        x = (x + 1e-6).log()
        kl = -x.mean(dim = -1)
        xent = F.nll_loss(x, y, reduction = 'none')
        return ((1 - self.smooth) * xent + self.smooth * kl).mean()
    

class SmoothNLLLoss(nn.Module):
    def __init__(self, smooth = 0.2):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, x, y):
        kl = -x.mean(dim = -1)
        xent = F.nll_loss(x, y, reduction = 'none')
        return ((1 - self.smooth) * xent + self.smooth * kl).mean()


class NLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        x = (x + 1e-6).log()
        y = y.long()
        return F.nll_loss(x, y)


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        y = y.long()
        return F.nll_loss(x, y)
    
    
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return F.binary_cross_entropy(x, y.float())
    
    
class SLLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1))
        x = x.float()
        y = y.float()
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
        ce = -(y * x.log()).sum(-1).mean()
        rce = -(x * y.log()).sum(-1).mean()
        return self.alpha * ce + self.beta * rce
        