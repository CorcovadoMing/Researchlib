from torch import nn


class _Name(nn.Module):
    def __init__(self, detach = False):
        super().__init__()
        self.detach = detach
    
    def forward(self, *x):
        if self.detach:
            x = [i.detach() for i in x]
        if len(x) == 1:
            return x[0]
        else:
            return x