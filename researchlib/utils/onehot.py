import torch
import numpy as np


def to_one_hot(x, length):
    x_one_hot = torch.zeros(x.size(0), length)
    x_one_hot[np.arange(x.size(0)), x] = 1.0
    return x_one_hot
