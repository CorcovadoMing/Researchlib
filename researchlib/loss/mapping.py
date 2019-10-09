import torch
from torch import nn
import torch.nn.functional as F
from .custom_loss import *
from ..metrics import Acc, BCEAcc, AUROC
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
    if loss_fn == 'nll':
        loss_fn = nll_loss
        default_metrics = [Acc()]

    elif loss_fn == 'nl':
        loss_fn = nl_loss
        default_metrics = [Acc()]

    elif loss_fn == 'smooth_nll':
        loss_fn = smooth_nll_loss
        default_metrics = [Acc()]

    elif loss_fn == 'bce':
        loss_fn = F.binary_cross_entropy
        default_metrics = [BCEAcc(), AUROC()]

    elif loss_fn == 'adaptive_robust':
        loss_fn = AdaptiveRobustLoss(1)
        default_metrics = []

    elif loss_fn == 'focal':
        loss_fn = FocalLoss()
        default_metrics = [Acc()]

    elif loss_fn == 'adaptive_focal':
        loss_fn = AdaptiveFocalLoss()
        default_metrics = [Acc()]

    elif loss_fn == 'margin':
        loss_fn = MarginLoss()
        default_metrics = [Acc()]

    elif loss_fn == 'mse' or loss_fn == 'l2':
        loss_fn = F.mse_loss
        default_metrics = []

    elif loss_fn == 'mae' or loss_fn == 'l1':
        loss_fn = F.l1_loss
        default_metrics = []

    elif loss_fn == 'huber':
        loss_fn = HuberLoss()
        default_metrics = []

    elif loss_fn == 'logcosh':
        loss_fn = LogCoshLoss
        default_metrics = []

    else:
        loss_fn = loss_fn
        default_metrics = []

    return loss_fn, default_metrics
