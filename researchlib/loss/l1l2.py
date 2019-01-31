from torch import nn
import torch.nn.functional as F

class L1L2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return F.mse_loss(x, y) + F.l1_loss(x, y)