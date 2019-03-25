from torch import nn

class Auxiliary(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.store = None
        
    def forward(self, x):
        # Store the result
        # Return the origin value
        self.store = self.f(x)
        return x 