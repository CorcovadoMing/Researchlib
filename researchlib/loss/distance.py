from torch import nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return F.l1_loss(x, y.float().view_as(x))
    
    
class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return F.mse_loss(x, y.float().view_as(x))