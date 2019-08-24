from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit
import copy


class _padding_shortcut(nn.Module):

    def __init__(self, in_dim, out_dim, pool_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pool_layer = nn.Sequential(*list(filter(None, [pool_layer])))

    def forward(self, x):
        x = self.pool_layer(x)
        if self.in_dim >= self.out_dim:
            return x[:, :self.out_dim]
        else:
            return torch.cat((x,
                              torch.autograd.Variable(
                                  torch.zeros(
                                      (x.size(0), self.out_dim - self.in_dim,
                                       x.size(2), x.size(3)),
                                      device=x.device))), 1)


class WideResBlock(_Block):
    '''
        TODO
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

        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        padding = 0 if is_transpose and self.do_pool else self._get_param(
            'padding', 1)
        kernel_size = 2 if is_transpose and self.do_pool else self._get_param(
            'kernel_size', 3)
        drop_layer = nn.Dropout(0.5) if self._get_param('dropout',
                                                        True) else None
        first_custom_kwargs = self._get_custom_kwargs({
            'kenel_size':
                kernel_size,
            'stride':
                stride,
            'padding':
                padding,
            'erased_activator':
                True if self.preact and erased_activator else False
        })
        second_custom_kwargs = self._get_custom_kwargs(
            {'erased_activator': True if not self.preact else False})

        conv_layers = [
            unit_fn(self.op, self.in_dim, self.out_dim, False, self.do_norm,
                    self.preact, **first_custom_kwargs), drop_layer,
            unit_fn(self.op, self.out_dim, self.out_dim, False, self.do_norm,
                    self.preact, **second_custom_kwargs),
            preact_final_norm_layer
        ]
        self.conv = nn.Sequential(*list(filter(None, conv_layers)))

        shortcut_type = self._get_param('shortcut', 'projection')
        if shortcut_type not in ['projection', 'padding']:
            raise ('Shortcut type is not supported')
        if shortcut_type == 'projection':
            shortcut_kernel_size = 2 if is_transpose and self.do_pool else 1
            if self.in_dim != self.out_dim or self.do_pool:
                custom_kwargs = self._get_custom_kwargs({
                    'kenel_size': shortcut_kernel_size,
                    'stride': stride
                })
                reduction_op = self.op(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=shortcut_kernel_size,
                    stride=stride)
            else:
                reduction_op = None
        elif shortcut_type == 'padding':
            pool_type = self._get_param('pool_type',
                                        'AvgPool')  # As paper's design
            pool_factor = self._get_param('pool_factor', 2)
            pool_layer = self._get_pool_layer(
                pool_type, pool_factor) if self.do_pool else None
            reduction_op = _padding_shortcut(self.in_dim, self.out_dim,
                                             pool_layer)

        self.shortcut = nn.Sequential(*list(filter(None, [reduction_op])))

        self.se = self._get_param('se', True)
        if self.se:
            self.se_branch = self._get_se_branch()
        self.shakedrop = self._get_param('shakeDrop', False)
        if self.shakedrop:
            self.shakedrop_branch = self._get_shake_drop_branch()

    def forward(self, x):
        _x = self.conv(x)
        if self.se:
            _x = _x * self.se_branch(_x)
        if self.shakedrop:
            _x = self.shakedrop_branch(_x)
        x = self.shortcut(x)
        return self.merge_layer(x + _x)
