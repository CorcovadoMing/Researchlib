from torch import nn
import torch

class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.delta = delta
    
    def forward(self, x, y):
        mae = torch.abs(y - x)
        loss = torch.where(
            mae < self.delta, 
            0.5 * (mae ** 2), 
            self.delta * mae - 0.5 * (self.delta ** 2)
            )
        return loss.mean()