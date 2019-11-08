from ...models import Builder

def _Subgraph(flow, in_node, out_node):
    return Builder.Graph(flow, in_node=in_node, out_node=out_node)