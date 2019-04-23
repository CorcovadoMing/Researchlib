import torch
from torch import nn
from .mapping import *

class EnsembleLoss(nn.Module):
    def __init__(self, fns, weights):
        super().__init__()
        self.fns = fns
        self.weights = weights

    def forward(self, x, y):
        loss = 0
        for i in range(len(self.fns)):
            loss += self.weights[i] * self.fns[i](x, y)
        return loss

def loss_ensemble(cfg):
    loss_fns = []
    loss_weights = []
    rl, kx, ky, dm = None, None, None, []
    for i in cfg:
        res = loss_mapping(i)
        loss_fns.append(res[0])
        rl = res[1]
        kx = res[2]
        ky = res[3]
        if res[4]:
            dm = res[4]
        loss_weights.append(cfg[i])
    
    return EnsembleLoss(loss_fns, loss_weights), rl, kx, ky, dm