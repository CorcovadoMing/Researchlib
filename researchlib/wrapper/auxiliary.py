from torch import nn


class _Auxiliary(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f
        self.store = None
        self.hook = f.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.store = output

    def forward(self, x):
        self.f(x)
        return x
