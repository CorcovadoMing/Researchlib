from .graph_builder import _Graph
from .seq_builder import _Seq
from .node import Node


class Builder(object):
    Graph = _Graph
    Seq = _Seq
