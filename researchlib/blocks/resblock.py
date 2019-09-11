from .template.block import _Block
from .template.shortcut import _padding_shortcut
from ..layers import layer
from torch import nn
import torch
from .unit import unit


class _ResBlock(_Block):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''

    def __postinit__(self):
        is_transpose = self._is_transpose()

        unit_fn = self._get_param('unit', unit.conv)
        erased_activator = self._get_param('erased_activator', False)
        activator_type = self._get_param('activator_type', 'ReLU')
        activator_layer = self._get_activator_layer(
            activator_type
        ) if not erased_activator and not self.preact else None
        self.merge_layer = nn.Sequential(*list(filter(None, [activator_layer])))

        norm_type = self._get_param('norm_type', 'BatchNorm')
        preact_final_norm_layer = self._get_norm_layer(
            norm_type, self.out_dim) if self.do_norm and self.preact else None

        blur = self._get_param('blur', False) and self.do_pool
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        kernel_size = self._get_param('kernel_size', 2) if is_transpose and self.do_pool else self._get_param('kernel_size', 3)
        padding = 0 if is_transpose and self.do_pool else self._get_param('padding', int((kernel_size - 1) / 2))
        
        first_custom_kwargs = self._get_custom_kwargs({
            'kernel_size': kernel_size,
            'stride': 1 if blur else stride,
            'padding': padding,
            'erased_activator': True if self.preact and erased_activator else False
        })
        
        second_custom_kwargs = self._get_custom_kwargs({'erased_activator': True if not self.preact else False})
        
        conv_layers = [
            unit_fn(self.op, self.in_dim, self.out_dim, False, self.do_norm, self.preact, **first_custom_kwargs),
            layer.Downsample(channels=self.out_dim, filt_size=3, stride=stride) if blur else None,
            unit_fn(self.op, self.out_dim, self.out_dim, False, self.do_norm, self.preact, **second_custom_kwargs),
            preact_final_norm_layer
        ]
        
        self.conv = nn.Sequential(*list(filter(None, conv_layers)))

        shortcut_type = self._get_param('shortcut', 'projection')
        if shortcut_type not in ['projection', 'padding']:
            raise ('Shortcut type is not supported')
        if shortcut_type == 'projection':
            shortcut_kernel_size = 2 if is_transpose and self.do_pool else 1
            shortcut_stride = 1 if blur else stride
            if self.in_dim != self.out_dim or self.do_pool:
                reduction_op = [self.op(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=shortcut_kernel_size,
                    stride=shortcut_stride)]
                if blur:
                    reduction_op.append(layer.Downsample(channels=self.out_dim, filt_size=3, stride=stride))
            else:
                reduction_op = [None]
        elif shortcut_type == 'padding':
            pool_type = self._get_param('pool_type', 'AvgPool')  # As paper's design
            pool_factor = self._get_param('pool_factor', 2)
            pool_layer = self._get_pool_layer(pool_type, pool_factor, self.in_dim) if self.do_pool else None
            reduction_op = [_padding_shortcut(self.in_dim, self.out_dim, pool_layer)]

        self.shortcut = nn.Sequential(*list(filter(None, reduction_op)))

        self.branch_attention = self._get_param('branch_attention')
        if self.branch_attention:
            self.attention_branch = self._get_attention_branch()
        self.shakedrop = self._get_param('shakedrop', False)
        if self.shakedrop:
            self.shakedrop_branch = self._get_shake_drop_branch()

    def forward(self, x):
        _x = self.conv(x)
        if self.branch_attention:
            _x = self.attention_branch(_x)
        if self.shakedrop:
            _x = self.shakedrop_branch(_x)
        x = self.shortcut(x)
        return self.merge_layer(x + _x)
