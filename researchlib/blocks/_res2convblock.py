from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
import math
import torch.nn.utils.spectral_norm as sn


class _inner_res2conv(nn.Module):

    def __init__(self, op_group, norm_group, act_layer, scales, ignore_last):
        super().__init__()
        self.op_group = nn.ModuleList(op_group)
        self.norm_group = nn.ModuleList(norm_group)
        self.act_layer = act_layer
        self.scales = scales
        self.ignore_last = ignore_last

    def forward(self, x):
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        rang = self.scales - 1 if self.ignore_last else self.scales
        for s in range(rang):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(
                    self.act_layer(self.norm_group[s - 1](self.op_group[s - 1](
                        xs[s]))))
            else:
                ys.append(
                    self.act_layer(self.norm_group[s - 1](
                        self.op_group[s - 1](xs[s] + ys[-1]))))
        if self.ignore_last:
            ys.append(xs[-1])
        out = torch.cat(ys, 1)
        return out


class Res2ConvBlock(_Block):
    '''
        Res2Net: A New Multi-scale Backbone Architecture
        https://arxiv.org/abs/1904.01169
    '''

    def __postinit__(self):
        # Parameters
        activator_type = self._get_param('actvator_type', 'ReLU')
        norm_type = self._get_param('norm_type', 'BatchNorm')
        pool_type = self._get_param('pool_type', 'MaxPool')
        pool_factor = self._get_param('pool_factor', 2)
        conv_kwargs = self._get_conv_kwargs()

        spectral_norm = self._get_param('sn', False)
        erased_activator = self._get_param('erased_activator', False)

        # Layers
        if conv_kwargs['kernel_size'] == 1:
            conv_layer = self.op(self.in_dim, self.out_dim, **conv_kwargs)
        else:
            scales = self._get_param('scales', 4)
            test_case = torch.chunk(torch.randn(self.in_dim), scales)
            group, dim, ignore_last = len(test_case), len(
                test_case[0]), len(test_case[-1]) != len(test_case[0])
            dim = math.ceil(self.in_dim / scales)
            norm_group = [
                self._get_norm_layer(norm_type, dim) for _ in range(group - 1)
            ]
            stride = conv_kwargs['stride']
            conv_kwargs['stride'] = 1
            op_group = [
                self.op(dim, dim, **conv_kwargs) for _ in range(group - 1)
            ]
            conv_layer = []
            if stride != conv_kwargs['stride']:
                conv_layer.append(
                    self.op(self.in_dim, self.in_dim, 1, stride=stride))
            conv_layer.append(
                _inner_res2conv(op_group, norm_group,
                                self._get_activator_layer(activator_type),
                                scales, ignore_last))
            if self.in_dim != self.out_dim:
                conv_layer.append(self.op(self.in_dim, self.out_dim, 1))
            conv_layer = nn.Sequential(*conv_layer)

        if spectral_norm:
            conv_layer = sn(conv_layer)
        activator_layer = self._get_activator_layer(
            activator_type) if not erased_activator else None
        pool_layer = self._get_pool_layer(pool_type,
                                          pool_factor) if self.do_pool else None
        norm_layer = self._get_norm_layer(norm_type) if self.do_norm else None

        if self.preact:
            self.layers = [norm_layer, activator_layer, conv_layer, pool_layer]
        else:
            self.layers = [conv_layer, norm_layer, activator_layer, pool_layer]
        self.layers = nn.Sequential(*list(filter(None, self.layers)))

    def forward(self, x):
        return self.layers(x)
