from torch import nn

class _Inject(nn.Module):
    def __init__(self, orig_f, inject_f):
        super().__init__()
        self.orig_f = orig_f
        self.inject_f = inject_f
    
    def forward(self, x):
        return self.inject_f(self.orig_f(x))

def inject_after(model, node, inject_f):
    target_node = model[node]
    model[node] = (_Inject(target_node[0], inject_f), *target_node[1:])