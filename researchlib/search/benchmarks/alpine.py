import torch


def Alpine(x):
    # Optimal is at 0
    f = (x * torch.sin(x)) + (0.1 * x).abs()
    return f.sum().abs()
