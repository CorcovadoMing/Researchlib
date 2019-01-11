from .torchqrnn import *
from torch import nn
import torch 

class QRNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, return_sequences=False, bidirection=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if bidirection:
            self.forward_f = QRNNLayer_(in_dim, out_dim)
            self.backward_f = QRNNLayer_(in_dim, out_dim)
        else:
            self.f = QRNNLayer_(in_dim, out_dim)
        self.bidirection = bidirection
        self.rs = return_sequences
    
    def forward(self, x):
        x = x.permute(1, 0, 2)
        if self.bidirection:
            x_f, _ = self.forward_f(x)
            x_b, _ = self.backward_f(x)
            x = torch.cat((x_f, x_b), dim=-1)
        else:
            x, _ = self.f(x)
        x = x.permute(1, 0, 2)
        
        if self.rs:
            return x
        else:
            return x[:, -1, :]