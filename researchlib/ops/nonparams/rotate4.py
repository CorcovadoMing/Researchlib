from torch import nn
import torch


r0 = lambda x: x
r90 = lambda x: x.transpose(-2, -1)
r180 = lambda x: x.flip(-2)
r270 = lambda x: x.transpose(-2, -1).flip(-1)

class _Rotation42d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _get_process_fn(self, angle):
        if angle == 0:
            return r0
        elif angle == 1:
            return r90
        elif angle == 2:
            return r180
        elif angle == 3:
            return r270
    
    def forward(self, x):
        label = torch.randint(0, 4, [x.size(0)]).to(x.device)
        for i in range(len(x)):
            x[i] = self._get_process_fn(label[i])(x[i])
        return x, label