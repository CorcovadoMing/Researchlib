import torch.nn.functional as F
import math


def LogCoshLoss(x, y):
    diff = y - x
    return (diff + F.softplus(-2. * diff) - math.log(2.)).mean()
