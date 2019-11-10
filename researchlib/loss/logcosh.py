import torch.nn.functional as F
from torch import nn
import math


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        diff = y - x
        return (diff + F.softplus(-2. * diff) - math.log(2.)).mean()
