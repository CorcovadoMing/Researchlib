from torch import nn
from ..wrapper import _MultiscaleOutput, _MultiscaleInput


class builder(nn.Module):
    # Static builder queue
    queue = []

    def __init__(self, nnlist):
        super().__init__()
        self.nnlist = nn.ModuleList(nnlist)

    def forward(self, x):
        for i in range(len(self.nnlist)):
            if type(self.nnlist[i]) == _MultiscaleOutput:
                x = self.nnlist[i](x)
                builder.queue.append(x)
            elif type(self.nnlist[i]) == _MultiscaleInput:
                x = self.nnlist[i](x, builder.queue.pop())
            else:
                x = self.nnlist[i](x)
        return x

    def num_params(self):
        return sum(
            p.numel() for p in self.nnlist.parameters() if p.requires_grad)
