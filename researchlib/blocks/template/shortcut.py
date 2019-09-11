from torch import nn
import torch


class _padding_shortcut(nn.Module):
    def __init__(self, in_dim, out_dim, pool_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pool_layer = nn.Sequential(*list(filter(None, [pool_layer])))

    def forward(self, x):
        x = self.pool_layer(x)
        if self.in_dim >= self.out_dim:
            return x[:, :self.out_dim]
        else:
            return torch.cat(
                (x,
                 torch.autograd.Variable(
                     torch.zeros(
                         (x.size(0), self.out_dim - self.in_dim, *x.shape[2:]),
                         device=x.device))), 1)