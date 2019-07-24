from torch import nn
from ..wrapper import MultiscaleOutput, MultiscaleInput


class builder(nn.Module):
    # Static builder queue
    queue = []

    def __init__(self, nnlist):
        super().__init__()
        self.nnlist = nn.ModuleList(nnlist)

    def forward(self, x):
        for i in range(len(self.nnlist)):
            if type(self.nnlist[i]) == MultiscaleOutput:
                x = self.nnlist[i](x)
                builder.queue.append(x)
            elif type(self.nnlist[i]) == MultiscaleInput:
                x = self.nnlist[i](x, builder.queue.pop())
            else:
                x = self.nnlist[i](x)
        return x
