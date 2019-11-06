from torch import nn


class _FlattenExcept(nn.Module):
    def __init__(self, axis = None):
        super().__init__()
        self.axis = axis
    
    def forward(self, x):
        if self.axis is None:
            return x.view(-1)
        else:
            collape = int(x.numel() / x.size(self.axis))
            return x.view(collape, x.size(self.axis))