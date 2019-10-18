import torch
from torch import nn
import torch.nn.functional as F
from .custom_loss import *
from .margin import MarginLoss
from torch.autograd import Variable
import numpy as np


def smooth_nll_loss(x, y, smoothing = 0.2):
    kl = -x.mean(dim=-1)
    xent = F.nll_loss(x, y, reduction='none')
    return ((1-smoothing) * xent + smoothing * kl).mean()
    
def nl_loss(x, y):
    y = y.squeeze().long()
    return -x[range(y.shape[0]), y].log().mean()

def nll_loss(x, y):
    y = y.squeeze().long()
    return F.nll_loss(x, y)

def loss_mapping(loss_fn):
    # Assign loss function
    mapping = {
        'nll': nll_loss,
        'nl': nl_loss,
        'smooth_nll': smooth_nll_loss,
        'bce': F.binary_cross_entropy,
        'adaptive_robust': AdaptiveRobustLoss(1), # need to fix dimensions
        'focal': FocalLoss(), # need to be refined
        'adaptive_focal': AdaptiveFocalLoss(), # Experimental
        'margin': MarginLoss(),
        'mse': F.mse_loss,
        'l2': F.mse_loss,
        'mae': F.l1_loss,
        'l1': F.l1_loss,
        'huber': HuberLoss(),
        'logcosh': LogCoshLoss
    }
    
    if type(loss_fn) == str:
        loss_fn = mapping[loss_fn]
    else:
        loss_fn = loss_fn
    return loss_fn
