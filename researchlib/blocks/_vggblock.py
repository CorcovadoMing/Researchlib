from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit


class VGGBlock(_Block):
    '''
    '''

    def __postinit__(self):
        is_transpose = self._is_transpose()
        unit_fn = self._get_param('unit', unit.conv)
        self.unit = unit_fn(self.op, self.in_dim, self.out_dim, self.do_pool,
                            self.do_norm, self.preact)

    def forward(self, x):
        return self.unit(x)
