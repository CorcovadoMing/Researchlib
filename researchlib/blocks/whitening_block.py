from .utils import get_conv_config
from ..utils import ParameterManager
import torch
from torch import nn


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
        self.inner_op = _op(in_dim, 27, kernel_size=3, padding=1, bias=False)
        self.inner_op.weight.requires_grad = False
    
    def clear_source(self):
        self.reset_parameters()
    
    def forward(self, x):
        if self.e1 is None or self.e2 is None:
            self.e1 = ParameterManager.get_buffer('e1', clear=False)
            self.e2 = ParameterManager.get_buffer('e2', clear=False)
            self.inner_op.weight.data = (self.e2 / torch.sqrt(self.e1 + 1e-4)[:, None, None, None]).to(x.device).to(x.dtype)
            self.inner_op.weight.requires_grad = False
        return self.unit(self.inner_op(x))
    
    def reset_parameters(self):
        self.e1 = None
        self.e2 = None