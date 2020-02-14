from ...loss import Loss
from ...ops import op
from ...models import Node, Optimize, Heads


def _IMSAT(num_cluster, backend, input_node, output_node='cluster_prob', model_node='model'):
    model = [
        backend(input_node, model_node),
        Node('cluster_head', Heads(num_cluster, reduce_type='avg'), model_node),
        Node(output_node, op.Softmax(-1), 'cluster_head'),
        Node('input_p', op.RPT(), input_node),
        Node('shared', ['model', 'cluster_head', output_node], 'input_p'),
        Optimize('sat', Loss.Cluster.IMSAT(), output_node, 'shared'),
    ]
    return model