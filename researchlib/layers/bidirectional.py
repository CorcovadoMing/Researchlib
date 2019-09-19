from torch import nn
import torch
import copy


class Bidirectional(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.forward_f = f
        self.backward_f = copy.deepcopy(f)

    def forward(self, x):
        x_forward = self.forward_f(x)
        x_backward = self.backward_f(x)
        return torch.cat((x_forward, x_backward), dim = -1)
