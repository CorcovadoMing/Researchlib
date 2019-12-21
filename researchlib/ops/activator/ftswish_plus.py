from torch import nn
import torch.nn.functional as F
import torch


class _FTSwishPlus(nn.Module):
    def __init__(self, threshold=-.25, mean_shift=-.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x): 
        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        x.sub_(self.mean_shift)
        return x