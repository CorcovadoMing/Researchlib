from torch import nn

class Reg(nn.Module):
    def __init__(self, f, group, get='weight', out_through=False):
        super().__init__()
        self.f = f
        self.get = get
        self.reg_store = None
        self.reg_group = group
        self.out = out_through
        
    def forward(self, x):
        out = self.f(x)
        if self.get == 'weight':
            self.reg_store = self.f.weight
        elif self.get == 'out':
            self.reg_store = out
        elif self.get == 'in':
            self.reg_stroe = x
        
        if self.out:
            return x
        else:
            return out
    