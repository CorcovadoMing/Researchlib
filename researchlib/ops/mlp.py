from torch import nn
    

class _MLP(nn.Module):
    def __init__(self, *layers, act=nn.ELU()):
        super().__init__()
        _layers = []
        for j in range(len(layers)-1):
            _act = act if j < len(layers)-2 else None
            _layers += [nn.Linear(layers[j], layers[j+1]), _act]
        self.net = nn.Sequential(*filter(None, _layers))
    
    def forward(self, x):
        return self.net(x)
        