import numpy as np
import scipy.stats as st
import torch
from torch import nn
import torch.nn.functional as F


def _gen_kernel(kernel_size=5, sigma=3):
    interval = (2*sigma+1.)/(kernel_size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernel_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel.astype(np.float32)


class _Gaussian2d(nn.Module):
    def __init__(self, in_dim, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_dim = in_dim
        self.prior = torch.from_numpy(_gen_kernel(kernel_size))[None, None, ...].repeat(in_dim, 1, 1, 1)
    
    def forward(self, x):
        return F.conv2d(x, self.prior, padding=int((self.kernel_size-1)/2), groups=self.in_dim)