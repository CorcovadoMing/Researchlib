from torch import nn

class _Block(nn.Module):
    def __init__(self, op, in_dim, out_dim, **kwargs):
        super().__init__()
        self.op = op
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kwargs = kwargs
    
    def forward(self, x):
        raise('Not implemented')