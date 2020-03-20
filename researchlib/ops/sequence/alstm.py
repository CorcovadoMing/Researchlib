from .torchalstm import *
from torch import nn
import torch


class _ALSTM(nn.Module):
    def __init__(self, in_dim, out_dim, adaptive_dim, return_sequences = False, bidirection = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if bidirection:
            self.forward_f = aLSTM(in_dim, out_dim, adaptive_dim, batch_first = True)
            self.backward_f = aLSTM(in_dim, out_dim, adaptive_dim, batch_first = True)
        else:
            self.f = aLSTM(in_dim, out_dim, adaptive_dim, batch_first = True)
        self.bidirection = bidirection
        self.rs = return_sequences

    def forward(self, x):
        # bs, ts, feature -> ts, bs, feature
        x = x.transpose(0, 1)
        if self.bidirection:
            ts = list(range(x.size(0)))
            x_f, _ = self.forward_f(x)
            x_b, _ = self.backward_f(x[ts, :, :])
            x = torch.cat((x_f, x_b), dim = -1)
        else:
            x, _ = self.f(x)
        x = x.transpose(0, 1)

        if self.rs:
            return x
        else:
            return x[:, -1, :]

