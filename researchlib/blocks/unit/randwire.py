from ..template.block import _Block
from ...layers import layer
from torch import nn
import torch
import math
import os
import networkx
import torch.nn.utils.spectral_norm as sn
import collections
from .conv import _conv
import random

__NodeTemplate__ = collections.namedtuple('__NodeTemplate__', ['id', 'inputs', 'type'])


class _randwire(_Block):
    def __postinit__(self):
        # Parameters
        pool_factor = self._get_param('pool_factor',
                                      2) if self.do_pool else self._get_param('stride', 1)
        conv_kwargs = self._get_conv_kwargs()
        num_nodes = self._get_param('num_nodes', 32)

        # Layers
        if conv_kwargs['kernel_size'] == 1:
            self.layers = _conv(
                self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact,
                **conv_kwargs
            )
        else:
            self.layers = _stage_block(self.in_dim, self.out_dim, num_nodes, pool_factor)

    def forward(self, x):
        return self.layers(x)


def _get_graph_info(graph):
    nodes = []
    input_nodes = []
    output_nodes = []
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        nodes.append(__NodeTemplate__(node, [n for n in tmp if n < node], type))
    return nodes, input_nodes, output_nodes


def _build_graph(nodes):
    return networkx.generators.random_graphs.connected_watts_strogatz_graph(
        nodes, 4, 0.75, tries = 200, seed = None
    )


class _depthwise_separable_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size = 3, stride = stride, padding = 1, groups = nin
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size = 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class _triplet_unit(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = _depthwise_separable_conv_3x3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


class _node_op(nn.Module):
    def __init__(self, node, in_dim, out_dim, stride):
        super().__init__()
        self.input_nums = len(node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        if node.type == 0:
            self.op = _triplet_unit(in_dim, out_dim, stride = stride)
        else:
            self.op = _triplet_unit(out_dim, out_dim, stride = 1)

    def forward(self, *input):
        if self.input_nums > 1:
            out = 0
            already_dropped = False
            for i in range(self.input_nums):
                # As paper described drop path regularization
                if random.random() > 0.1 or already_dropped:
                    out += self.sigmoid(self.mean_weight[i]) * input[i]
                else:
                    already_dropped = True
        else:
            out = input[0]
        out = self.op(out)
        return out


class _stage_block(nn.Module):
    def __init__(self, inplanes, outplanes, num_nodes, stride):
        super().__init__()
        graph = _build_graph(num_nodes)
        self.nodes, self.input_nodes, self.output_nodes = _get_graph_info(graph)
        self.nodeop = nn.ModuleList()
        for node in self.nodes:
            self.nodeop.append(_node_op(node, inplanes, outplanes, stride))

    def forward(self, x):
        results = {}
        for id in self.input_nodes:
            results[id] = self.nodeop[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                results[id] = self.nodeop[id](*[results[_id] for _id in node.inputs])
        result = results[self.output_nodes[0]]
        for idx, id in enumerate(self.output_nodes):
            if idx > 0:
                result = result + results[id]
        result = result / len(self.output_nodes)
        return result
