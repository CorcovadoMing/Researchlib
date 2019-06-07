from torch import nn
from ..wrapper import MultiscaleOutput
        
class builder(nn.Module):
    def __init__(self, nnlist, input_multiscale=False, return_multiscale=False):
        super().__init__()
        self.return_multiscale = return_multiscale
        self.input_multiscale = input_multiscale
        self.nnlist = nn.ModuleList(nnlist)
    
    def forward(self, x):
        if self.return_multiscale:
            out = []
            final_append=False
            for i in range(len(self.nnlist)):
                x = self.nnlist[i](x)
                if type(self.nnlist[i]) == MultiscaleOutput:
                    out.append(x)
                    final_append=True
                else:
                    final_append=False
            if final_append:
                return x, out[:-1]
            else:
                return x, out
        else:
            for i in range(len(self.nnlist)):
                x = self.nnlist[i](x)
            return x
        