import torch
from torch import nn
import math
import torch.nn.functional as F


class _NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def forward(self, x):
        if self.training:
            rand_in = self._f(
                torch.randn(1, self.in_features, device=self.weight.device))
            rand_out = self._f(
                torch.randn(self.out_features, 1, device=self.weight.device))
            epsilon_w = torch.matmul(rand_out, rand_in)
            epsilon_b = rand_out.squeeze()

            w = self.weight + self.sigma_w * epsilon_w
            b = self.bias + self.sigma_b * epsilon_b
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.weight, self.bias)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))
