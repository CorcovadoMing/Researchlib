from torch import nn
import torch


class QuantileLoss(nn.Module):
    '''
        Quantile Loss gives different weighting on higher/lower prediction by gamma.
        gamma = 0.5 equals MAE
        gamma < 0.5 gives more panalty on higer prediction
        gamma > 0.5 gives more panalty on lower prediction
    '''
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, x, y):
        mae = torch.abs(y - x)
        loss = torch.where(x > y, (1 - self.gamma) * mae, self.gamma * mae)
        return loss.mean()
