from collections import namedtuple
import torch


class Acc(namedtuple('Acc', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        x = x.view(-1)
        if y.numel() != x.size(0):
            y = y.argmax(-1)
        y = y.view(-1)
        return x.eq(y).float().mean()

    
class FusedAcc(namedtuple('FusedAcc', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        _, pred = torch.max(x, 1)
        pred = pred.view(-1)
        if y.numel() != pred.size(0):
            y = y.argmax(-1)
        y = y.view(-1)
        return pred.eq(y).float().mean()