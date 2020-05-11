#
# Training Deep Neural Networks on Noisy Labels with Bootstrapping
# http://www-personal.umich.edu/~reedscot/bootstrap.pdf
#

import torch
from torch.nn import Module
import torch.nn.functional as F


class SoftBootstrappingNLLoss(Module):
    """
    Loss(x, y) = - (beta * y + (1 - beta) * x) * log(x)
    """
    def __init__(self, beta=0.8, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1)).to(x.dtype)
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
            
        loss = -((self.beta * y + (1.0 - self.beta) * x) * x.log()) 

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class HardBootstrappingNLLoss(Module):
    """
    Loss(x, y) = - (beta * y + (1 - beta) * z) * log(x)
    where z = argmax(x)
    """
    def __init__(self, beta=0.8, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, x, y):
        if y.shape != x.shape:
            y = F.one_hot(y, x.size(1)).to(x.dtype)
        x = torch.clamp(x, min=1e-6, max=1.0)
        y = torch.clamp(y, min=1e-4, max=1.0)
        
        z = F.one_hot(x.argmax(-1), x.size(1))
        loss = -((self.beta * y + (1.0 - self.beta) * z) * x.log()) 

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss