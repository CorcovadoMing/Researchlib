from torch import nn

class Split(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, x):
        self.f(x)
        return x