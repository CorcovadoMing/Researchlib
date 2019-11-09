import torch

def _Flip(x, dim=[-1]):
    return torch.flip(x, dim)