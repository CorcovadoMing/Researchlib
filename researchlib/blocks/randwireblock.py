import networkx as nx
from .utils import get_conv_config, get_config
from ..utils import ParameterManager
from ..ops import op
from torch import nn
import torch


class _RandWireNodeOp(nn.Module):
    def __init__(self, num_input, unit_factory):
        super().__init__()
        self.num_input = num_input
        if num_input > 1:
            self.weight = nn.Parameter(torch.ones(num_input))
            self.reg_drop = nn.Dropout(0.1)
            self.only_pos = nn.ReLU(inplace=True)
        self.unit_factory = unit_factory()
    
    def forward(self, *x):
        x = torch.stack(x)
        if self.num_input > 1:
            weighting = self.reg_drop(self.weight)
            weighting = self.only_pos(weighting)
            x = torch.einsum('fnchw,f -> nchw', x, weighting)
        else:
            x = x.sum(0)
        return self.unit_factory(x)

    
def _RandWireBlock(prefix, _unit, _op, in_dim, out_dim, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    N = parameter_manager.get_param('N', 32)
    K = parameter_manager.get_param('K', 4)
    P = parameter_manager.get_param('P', 0.75)
    
    config = get_config(prefix, _unit, _op, in_dim, out_dim, parameter_manager)
    pool_kwargs = get_conv_config()
    pool_kwargs.update(**kwargs)
    pool_kwargs.update(do_share_banks=config.do_share_banks,
                       do_pool=False,
                       stride=2 if kwargs['do_pool'] else 1,
                       randwire=True)
    
    regular_kwargs = get_conv_config()
    regular_kwargs.update(**kwargs)
    regular_kwargs.update(do_share_banks=config.do_share_banks,
                          do_pool=False,
                          randwire=True)
    
    pool_unit_factory = lambda: config._unit(config.prefix, config._op, config.in_dim, config.out_dim, **pool_kwargs)
    regular_unit_factory = lambda: config._unit(config.prefix, config._op, config.out_dim, config.out_dim, **regular_kwargs)
    
    graph = nx.random_graphs.connected_watts_strogatz_graph(N, K, P)
    
    topo = {}
    terminal_index = []

    for cur_index in graph:
        topo[cur_index] = [[], []]
        for i in graph.edges(cur_index):
            _, suc = i
            if suc > cur_index:
                topo[cur_index][1].append(suc)
            else:
                topo[cur_index][0].append(suc)

        if not len(topo[cur_index][1]):
            terminal_index.append(cur_index)

        if not len(topo[cur_index][0]) and cur_index != 0:
            topo[cur_index][0].append(0)
            topo[0][1].append(cur_index)
        
    flow = {}
    for i in topo:
        if i == 0:
            inputs = [f'{prefix}_input']
            flow[f'{prefix}_{i}'] = (_RandWireNodeOp(len(inputs), pool_unit_factory), inputs)
        else:
            inputs = list(map(lambda x: f'{prefix}_{x}', topo[i][0]))
            flow[f'{prefix}_{i}'] = (_RandWireNodeOp(len(inputs), regular_unit_factory), inputs)
    
    final_inputs = list(map(lambda x: f'{prefix}_{x}', terminal_index))
    flow[f'{prefix}_output'] = (_RandWireNodeOp(len(final_inputs), regular_unit_factory), final_inputs)
    
    
    return op.Subgraph(flow, in_node=f'{prefix}_input', out_node=f'{prefix}_output')
            