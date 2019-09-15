from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit


class _DAWNBlock(_Block):
    '''
        https://myrtle.ai/how-to-train-your-resnet-4-architecture/
    '''

    def __postinit__(self):
        is_transpose = self._is_transpose()
        unit_fn = self._get_param('unit', unit.conv)
        self.stem = unit_fn(self.op, self.in_dim, self.out_dim, self.do_pool,
                            self.do_norm, False, **self._get_custom_kwargs())
        self.res1 = unit_fn(self.op, self.out_dim, self.out_dim, False,
                            self.do_norm, False, **self._get_custom_kwargs())
        self.res2 = unit_fn(self.op, self.out_dim, self.out_dim, False,
                            self.do_norm, False, **self._get_custom_kwargs())
        self.branch_attention = self._get_param('branch_attention')
        if self.branch_attention:
            self.attention_branch = self._get_attention_branch()

    def forward(self, x):
        x = self.stem(x)
        _x = self.res1(x)
        _x = self.res2(_x)
        if self.branch_attention:
            _x = self.attention_branch(_x)
        return x + _x
