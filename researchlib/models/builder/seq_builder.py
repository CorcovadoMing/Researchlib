from torch import nn
import torch
from .graph_builder import _Graph


def _Seq(*models):
    model_list = []
    for i in models:
        if type(i) == list:
            model_list += i
        else:
            model_list.append(i)
    
    flow = {str(i): (j, ['input'] if i == 0 else [str(i-1)]) for i, j in enumerate(model_list)}
    return _Graph(flow, in_node='input', out_node=str(len(model_list)-1))