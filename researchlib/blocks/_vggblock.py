from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit
import copy


class VGGBlock(_Block):
    '''
    '''

    def __postinit__(self):
        is_transpose = self._is_transpose()
        unit_fn = self._get_param('unit', unit.conv)
        self.unit = unit_fn(self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact)
        self.se = self._get_param('se', True)
        if self.se: self.se_branch = self._get_se_branch()

    def forward(self, x):
        x = self.unit(x)
        if self.se: x = x * self.se_branch(x)
        return x
