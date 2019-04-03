import torch
from torch import nn
from ...utils import *

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x, y):
        x = self.softmax(x)
        pt = x[torch.arange(x.size(0)), y]
        loss = -1 * self.alpha * ((1 - pt) ** self.gamma) * pt.log()
        return loss.mean()

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.ones(10).float().cuda()
        self.softmax = nn.Softmax(-1)
        
    def update_alpha(self, alpha_p):
        self.alpha = alpha_p
        self.alpha /= (3 * self.alpha.std())
        self.alpha += (1 - self.alpha.mean())
        
        print(self.alpha)
        print(self.alpha.mean(), self.alpha.std())
        
    def forward(self, x, y):
        x = self.softmax(x)
        pt = x[torch.arange(x.size(0)), y]
        
        alpha = self.alpha.repeat(x.size(0)).view(x.size(0), -1)
        alpha = alpha[torch.arange(x.size(0)), y]
        
        loss = -1 * alpha * ((1 - pt) ** self.gamma) * pt.log()
        return loss.mean()
