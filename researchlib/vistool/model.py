import torch
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from torch.autograd import Variable
from IPython.display import display
import hiddenlayer as hl


class _Model:
    def __init__(self):
        pass

    def _computation_graph(self, var, params):
        """ Produces Graphviz representation of PyTorch autograd graph

        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function

        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        param_map = {id(v): k for k, v in params.items()}
        node_attr = dict(
            style = 'filled',
            shape = 'box',
            align = 'left',
            fontsize = '12',
            ranksep = '0.1',
            height = '0.2'
        )

        dot = Digraph(node_attr = node_attr, graph_attr = dict(size = "200,200"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor = 'orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor = 'lightblue')
                else:
                    dot.node(str(id(var)), str(type(var).__name__).replace('Backward', ''))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        display(dot)

    def _highlevel(self, model, trial):
        transforms = [
            hl.transforms.Rename(op = 'ATen', to = 'Norm'),
            hl.transforms.Fold("Norm > Elu > Conv", "PreActConvblock"),
            hl.transforms.Fold("BatchNorm > Elu > Conv", "PreActConvblock"),
            hl.transforms.Fold("BatchNorm > Relu > Conv", "PreActConvblock"),
            hl.transforms.Fold("Conv > BatchNorm > Relu", "Convblock"),
            hl.transforms.Fold("Conv > BatchNorm > Elu", "Convblock"),
            hl.transforms.FoldDuplicates(),
        ]
        hl_graph = hl.build_graph(model, trial, transforms = transforms)
        hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
        display(hl_graph)

    def computation_graph(self, model, input_shape):
        inputs = torch.randn(1, *input_shape)
        y = model(Variable(inputs))
        self._computation_graph(y, dict(model.named_parameters()))

    def highlevel(self, model, input_shape):
        inputs = torch.randn(1, *input_shape)
        self._highlevel(model, inputs)


#     def viewer(self, model, input_shape):
#         _ = interact(self._browser_ui, index=range(min(100, len(self.data))), continuous_update=False)
