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

    
class FusedAcc(namedtuple('FusedAcc', ('topk'))):
    def __call__(self, x, y):
        with torch.no_grad():
            x, y = x.detach(), y.detach()
            if y.numel() != x.size(0):
                y = y.argmax(-1)
            y = y.view(-1)
            _, pred = x.topk(self.topk, -1)
            pred = pred.t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))
            correct_k = correct[:self.topk].view(-1).float().sum(0)
            return correct_k.div_(x.size(0))
        