import numpy as np
import scipy.signal
import torch


def _discount(arr, coeff=0.99):
    if type(arr) == torch.Tensor:
        arr = arr.detach().cpu().numpy()
    return np.array(scipy.signal.lfilter([1], [1, -coeff], arr[::-1], axis=0)[::-1], dtype=np.float32)
