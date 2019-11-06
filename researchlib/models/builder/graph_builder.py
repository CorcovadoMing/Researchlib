from torch import nn
from graphviz import Digraph

#====================================================================================================

def split(path, sep = '/'):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def find_merge_point(edges, target):
    for i, j in enumerate(edges):
        if j[0] == target:
            return i

def make_dot_graph(nodes, edges):
    g = Digraph('G')
    subgraph = {}
    for name, attr in nodes:
        if type(attr) == _Graph:
            subgraph[name] = DotGraph(attr.graph, dryrun=True).edges
    
    for key in subgraph:
        merge_point = find_merge_point(edges, key)
        edges[merge_point-1] = (edges[merge_point-1][0], subgraph[key][0][1], {}) 
        edges[merge_point:merge_point+1] = subgraph[key][1:] + [(subgraph[key][-1][1], 
                                                             edges[merge_point+1][0], 
                                                             {'dryrun': False})]
        with g.subgraph(name='cluster_0') as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.update(style='filled', color='white')
            sub_edges = [(i[0], i[1]) for i in subgraph[key][1:]]
            c.edges(sub_edges)
            c.attr(label=key)
    
    for src, dst, attr in edges:
        ignore = attr['dryrun'] if 'dryrun' in attr else False
        if not ignore:
            g.edge(src, dst)
    return g

class DotGraph():
    def __init__(self, graph, dryrun=False):
        self.nodes = [(k, v) for k, (v,_) in graph.items()]
        self.edges = [(src, dst, {'dryrun': dryrun}) for dst, (_, inputs) in graph.items() for src in inputs]
        if not dryrun:
            self.g = make_dot_graph(self.nodes, self.edges)
            
#====================================================================================================


has_inputs = lambda node: type(node) is tuple

def path_iter(nested_dict, pfx = ()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)


def normpath(path, sep = '/'):
    #simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..': parts.pop()
        elif p.startswith(sep): parts = [p]
        else: parts.append(p)
    return sep.join(parts)


def pipeline(net, sep = '/'):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1])))
            for (path, node) in path_iter(net)]


def build_graph(net, sep = '/'):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join(
        (path, '..', rel_path)
    )) if isinstance(rel_path, str) else flattened[idx + rel_path][0]
    return {
        path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]])
        for idx, (path, node) in enumerate(flattened)
    }


class _Graph(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items(): 
            setattr(self, path.replace('/', '_'), val)

    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            if k not in outputs:
                inp = []
                for x in ins:
                    # has index
                    if ':' in x:
                        x, index = x.split(':')
                        index = int(index)
                    else:
                        index = -1
                    
                    # Multiple Input
                    if type(outputs[x]) == tuple:
                        inp += list(outputs[x])
                    else:
                        inp.append(outputs[x])
                    
                    # has index 
                    if index >= 0:
                        inp = [inp[index]]

                outputs[k] = node(*inp)
        return outputs

    def show(self):
        return DotGraph(self.graph).g
        
