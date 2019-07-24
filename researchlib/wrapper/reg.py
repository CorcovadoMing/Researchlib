from torch import nn


class Reg(nn.Module):
    def __init__(self, f, group, get='weight'):
        super().__init__()
        self.f = f
        self.get = get
        self.reg_store = []
        self.reg_group = group
        self.hook = f.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if module.training:
            if self.get == 'weight':
                self.reg_store.append(module.weight)
            elif self.get == 'out':
                self.reg_store.append(output)
            elif self.get == 'in':
                self.reg_store.append(input)

    def forward(self, x):
        return self.f(x)
