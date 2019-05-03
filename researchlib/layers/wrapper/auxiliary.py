from torch import nn

class Auxiliary(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.store = []
        self.hook = f.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if module.training:
            self.store.append(output)

    def forward(self, x):
        self.f(x)
        return x