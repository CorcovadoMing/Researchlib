from torch import nn

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)