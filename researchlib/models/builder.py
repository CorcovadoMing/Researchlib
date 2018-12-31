import torch

def builder(nnlist):
    return torch.nn.Sequential(*nnlist)