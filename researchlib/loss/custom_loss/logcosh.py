import torch

def LogCoshLoss(x, y):
    return (torch.cosh(y - x)).log().mean()
