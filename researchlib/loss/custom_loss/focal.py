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
        loss = -1 * self.alpha * (1 + (1 - pt)) ** self.gamma * pt.log()
        return loss.mean()

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.softmax = nn.Softmax(-1)
        
    def update_alpha(self, alpha_p):
        self.alpha = alpha_p
        
    def forward(self, x, y):
        x = self.softmax(x)
        classes = x.size(1)
        y_p = to_one_hot(y, classes)
        y_onehot = y_p[torch.arange(x.size(0)), y].cuda()
        pt = x[torch.arange(x.size(0)), y]
        loss = -1 * self.alpha * ((y_onehot - pt) ** self.gamma) * pt.log()
        return loss.mean()
