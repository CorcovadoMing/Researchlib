from torch import nn
import torch.nn.functional as F


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
        y = y.long()
        return -x[range(y.shape[0]), y].log().mean()


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        y = y.long()
        return F.nll_loss(x, y)