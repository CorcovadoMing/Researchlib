from torch import nn
from graphviz import Digraph

#====================================================================================================

def split(path, sep = '/'):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def prefix_name(name, prefix):
    if ':' in name:
        name = name[:name.index(':')]
    if name[0] == '*':
        name = name[1:]
    if len(prefix) > 0:
        return str(prefix) + '_' + str(name)
    else:
        return name

def find_merge_endpoint(edges, target):
    for i, j in enumerate(edges):
        if j[1] == target:
            return i

def find_merge_startpoint(edges, target):
    for i, j in enumerate(edges):
        if j[0] == target:
            return i

def make_dot_graph(nodes, edges):
    g = Digraph('G')
    subgraph = {}
    for name, attr in nodes:
        if type(attr) == _Graph:
            subgraph[name] = DotGraph(attr.graph, dryrun=True, prefix=name).edges
    
    for key in subgraph:
        merge_point = find_merge_endpoint(edges, key)
        edges[merge_point] = (edges[merge_point][0], subgraph[key][0][1], {}) 
        with g.subgraph(name='cluster_'+key) as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.update(style='filled', color='white')
            sub_edges = [(i[0], i[1]) for i in subgraph[key][1:]]
            c.edges(sub_edges)
            c.attr(label=key)
        merge_point = find_merge_startpoint(edges, key)
        edges[merge_point] = (subgraph[key][-1][1], edges[merge_point][1], {})
    
    for src, dst, attr in edges:
        ignore = attr['dryrun'] if 'dryrun' in attr else False
        if not ignore:
            g.edge(src, dst)
    return g

class DotGraph():
    def __init__(self, graph, dryrun=False, prefix=''):
        self.nodes = [(k, v) for k, (v,_) in graph.items()]
        self.edges = [(prefix_name(src, prefix), prefix_name(dst, prefix), {'dryrun': dryrun}) for dst, (_, inputs) in graph.items() for src in inputs]
        if not dryrun:
            self.g = make_dot_graph(self.nodes, self.edges)
            self.g.attr(rankdir='LR')
            
            
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
    
    result = {
        path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]])
        for idx, (path, node) in enumerate(flattened)
    }
    
    # Post-fix
    from ...ops import op
    for k, v in result.items():
        if type(v[0]) == op.Source:
            result[k] = (v[0], ['phase'])
    
    return result


class _Graph(nn.Module):
    def __init__(self, *net, in_node='x', out_node=None, _seq_auto_build=False):
        super().__init__()
        
        self.visualize_nodes = []
        self.optimize_nodes = []
        self.monitor_nodes = []
        self.checkpoint_mode = None
        self.checkpoint_node = None
        self.checkpoint_state = None
        
        if len(net) == 1 and type(net[0]) == dict:
            self.graph = build_graph(net[0])
        else:
            self.graph = build_graph(self._expand_net(net))
        
        self.in_node = in_node
        self.out_node = out_node
        self._seq_auto_build = _seq_auto_build
        self.train_mode = True
        
        for path, (val, _) in self.graph.items(): 
            setattr(self, path.replace('/', '_'), val)
    
    def _parse_node(self, cur_node):
        node, node_type = cur_node
        if node_type == '__VISUAL__':
            self.visualize_nodes += list(node.keys())
        elif node_type == '__OPTIMIZE__':
            self.optimize_nodes += list(node.keys())
        elif node_type == '__MONITOR_MAX__':
            self.monitor_nodes += list(node.keys())
            self.checkpoint_mode = max
            self.checkpoint_node = list(node.keys())[0]
        elif node_type == '__MONITOR_MIN__':
            self.monitor_nodes += list(node.keys())
            self.checkpoint_mode = min
            self.checkpoint_node = list(node.keys())[0]
        elif node_type == '__MONITOR__':
            self.monitor_nodes += list(node.keys())
        return node
    
    def _expand_net(self, net):
        result = {}
        for i in net:
            if (type(i) == tuple or type(i) == list) and len(i) == 2:
                result.update(self._parse_node(i))
            elif (type(i) == tuple or type(i) == list):
                for j in i:
                    result.update(self._parse_node(j))
            elif type(i) == dict:
                result.update(i)
            else:
                raise ValueError(f'Unknown node type {type(i)}')
        return result
    
    
    def prepare_inp(self, ins, outputs):
        inp = []
        for x in ins:
            # has index
            if ':' in x:
                x, index = x.split(':')
                index = int(index)
            else:
                index = -1

            if '$' in x:
                x, key = x.split('$')
                target_node = self.graph[x][0]
                if len(key) == 0:
                    cand = target_node.parameters()
                else:
                    cand = target_node._parameters[key]
            else:
                cand = outputs[x]

            # has index 
            if index >= 0:
                cand = cand[index]
            
            inp.append(cand)
        return inp
    
                
    def forward(self, inputs):
        if type(inputs) != dict:
            inputs = {self.in_node: inputs}
        
        outputs = inputs
        for k, (node, inp) in self.graph.items():
            
            # Test-time operation
            if k[0] == '*':
                k = k[1:]
                if self.train_mode:
                    outputs[k] = None
                    continue
                    
            # Shared operation
            if type(node) == list or type(node) == tuple:
                node = [self.graph[i][0] for i in node]
            
            if k not in outputs:
                inp = self.prepare_inp(inp, outputs)
                
                # No cache in pipeline of shared models
                if type(node) == list:
                    for op in node:
                        inp = op(*inp)
                        inp = [inp]
                    outputs[k] = inp[0]
                else:
                    outputs[k] = node(*inp)
                    if type(node) == _Graph:
                        if node._seq_auto_build:
                            key_list = list(node.outputs.keys())
                            for i in key_list:
                                new_key = f'{k}/{i}'
                                node.outputs[new_key] = node.outputs.pop(i)
                        outputs.update(node.outputs)
                
        self.outputs = outputs
        if self.out_node is not None:
            return outputs[self.out_node]
        else:
            return outputs

    def show(self):
        return DotGraph(self.graph).g
        
