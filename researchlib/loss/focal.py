import torch
from torch import nn

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
