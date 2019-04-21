from torch import nn
import numpy as np
import torch

class PositionEncoding(nn.Module):
    def __init__(self, dk=512):
        super().__init__()
        self.encode_map = None
        self.dk = dk
    
    def _gen_map(self, pos, length, device):
        self.encode_map = np.zeros((length, pos))
        for p in range(pos):
            for l in range(length):
                if l % 2:
                    self.encode_map[l, p] = np.cos(p/(10000**((2*l)/self.dk)))
                else:
                    self.encode_map[l, p] = np.sin(p/(10000**((2*l)/self.dk)))
        self.encode_map = torch.from_numpy(self.encode_map).to(device).float()
        self.encode_map.require_grad = False

    def forward(self, x):
        if type(self.encode_map) == type(None):
            self._gen_map(x.size(2), x.size(1), x.device)
        elif self.encode_map.size(0) != x.size(1) or self.encode_map.size(1) != x.size(2):
            self._gen_map(x.size(2), x.size(1), x.device)
        return x + self.encode_map