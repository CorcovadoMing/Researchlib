from torch import nn
    

class _MLP(nn.Module):
    def __init__(self, *layers, act=nn.ELU(), input_flatten=False):
        super().__init__()
        _layers = []
        if input_flatten:
            _layers.append(nn.Flatten())
        for j in range(len(layers)-1):
            _act = act if j < len(layers)-2 else None
            _layers += [nn.Linear(layers[j], layers[j+1]), _act]
        self.net = nn.Sequential(*filter(None, _layers))
    
    def forward(self, x):
        return self.net(x)
        