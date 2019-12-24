from torch import nn


class _Name(nn.Module):
    def __init__(self, detach = False, select_index = None):
        super().__init__()
        self.detach = detach
        self.select_index = select_index
    
    def forward(self, *x):
        if self.detach:
            x = [i.detach() for i in x]
        if self.select_index is not None:
            x = [i[self.select_index] for i in x]
        if len(x) == 1:
            return x[0]
        else:
            return x