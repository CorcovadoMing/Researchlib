from torch import nn


class OHEM(nn.Module):
    def __init__(self, f, ratio=0.7):
        super().__init__()
        self.f = f
        self.ratio = ratio
        if self.f.reduction != 'none':
            self.f.reduction = 'none'
    
    def forward(self, x, y):
        loss = self.f(x, y)
        hes = int(loss.size(0) * self.ratio)
        loss, _ = loss.topk(hes)
        return loss.mean()