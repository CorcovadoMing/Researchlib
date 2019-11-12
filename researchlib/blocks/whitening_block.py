from .utils import get_conv_config
from ..utils import ParameterManager
import torch
from torch import nn


def _whitening_filter(_op, in_dim, Λ, V, device, dtype, eps=1e-2):
    filt = _op(in_dim, 27, kernel_size=3, padding=1, bias=False).to(device).to(dtype)
    filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None]).to(device).to(dtype)
    filt.weight.requires_grad = False
    return filt

class _WhiteningBlock(nn.Module):
    def __init__(self, prefix, _unit, _op, in_dim, out_dim, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.op = _op
        new_kwargs = get_conv_config()
        new_kwargs.update(**kwargs)
        new_kwargs.update(kernel_size=1, padding=0)
        self.unit = _unit(prefix, _op, 27, out_dim, **new_kwargs)
        self.e1 = None
        self.e2 = None
    
    def forward(self, x):
        if self.e1 is None or self.e2 is None:
            self.e1 = ParameterManager.get_buffer('e1', clear=False)
            self.e2 = ParameterManager.get_buffer('e2', clear=False)
        return self.unit(_whitening_filter(self.op, self.in_dim, self.e1, self.e2, device=x.device, dtype=x.dtype)(x))