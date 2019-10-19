from torch import nn

class _MultiApply(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *x):
        return [self.model(i) for i in x]