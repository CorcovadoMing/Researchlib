from .template.block import _Block
from .unit import unit
from researchlib.utils import ParameterManager
import torch


def _whitening_filter(op, in_dim, Λ, V, device, dtype, eps=1e-2):
    filt = op(in_dim, 27, kernel_size=(3, 3), padding=(1,1), bias=False).to(device).to(dtype)
    filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None]).to(device).to(dtype)
    filt.weight.requires_grad = False
    return filt

class _WhiteningBlock(_Block):
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        self.unit = unit_fn(
            self.op, 27, self.out_dim, self.do_pool, self.do_norm, self.preact,
            **self._get_custom_kwargs({'kernel_size': 1, 'padding': 0})
        )
        self.e1 = None
        self.e2 = None
    
    def forward(self, x):
        if self.e1 is None or self.e2 is None:
            self.e1 = ParameterManager.get_buffer('e1', clear=False)
            self.e2 = ParameterManager.get_buffer('e2', clear=False)
        return self.unit(_whitening_filter(self.op, self.in_dim, self.e1, self.e2, device=x.device, dtype=x.dtype)(x))