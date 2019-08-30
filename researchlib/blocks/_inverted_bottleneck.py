from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit


class InvertedBottleneckBlock(_Block):
    '''
        InvertedBottleneckBlock didn't support preact settings yet
    '''

    def __postinit__(self):
        is_transpose = self._is_transpose()
        unit_fn = self._get_param('unit', unit.conv)

        # Layers
        mb_factor = self._get_param('mb_factor', 5)
        hidden_size = self.out_dim * mb_factor
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        kernel_size = 2 if is_transpose and self.do_pool else self._get_param(
            'kernel_size', 3)
        padding = 0 if is_transpose and self.do_pool else self._get_param(
            'padding', int((kernel_size - 1) / 2))
        first_custom_kwargs = self._get_custom_kwargs({
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'erased_activator': False
        })
        second_custom_kwargs = self._get_custom_kwargs({
            'kenel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'erased_activator': False,
            'groups': hidden_size
        })
        third_custom_kwargs = self._get_custom_kwargs({
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'erased_activator': True
        })
        pre_conv_layers = [
            unit_fn(self.op, self.in_dim, hidden_size, False, self.do_norm,
                    False, **first_custom_kwargs),
            unit_fn(self.op, hidden_size, hidden_size, False, self.do_norm,
                    False, **second_custom_kwargs),
        ]
        self.pre_conv = nn.Sequential(*list(filter(None, pre_conv_layers)))
        self.post_conv = unit_fn(self.op, hidden_size, self.out_dim, False,
                                 self.do_norm, False, **third_custom_kwargs)
        self.need_shortcut = not self.do_pool and self.in_dim == self.out_dim

        self.branch_attention = self._get_param('branch_attention')
        if self.branch_attention:
            self.attention_branch = self._get_attention_branch(dim=hidden_size)
        self.shakedrop = self._get_param('shakeDrop', False)
        if self.shakedrop:
            self.shakedrop_branch = self._get_shake_drop_branch()

    def forward(self, x):
        _x = self.pre_conv(x)
        if self.branch_attention:
            _x  = self.attention_branch(_x)
        _x = self.post_conv(_x)

        if self.need_shortcut:
            if self.shakedrop:
                _x = self.shakedrop_branch(_x)
            return x + _x
        else:
            return _x
