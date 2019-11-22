from torch import nn
from ...utils import Annealer


class _Anneal(nn.Module):
    def __init__(self, name, srange, anneal_type='linear'):
        super().__init__()
        if anneal_type == 'cosine':
            anneal_policy = Annealer.Cosine
        elif anneal_type == 'linear':
            anneal_policy = Annealer.Linear
        self.name = name
        Annealer.set_trace(name, None, srange, 'epoch', anneal_policy)
        
    def forward(self, x):
        if self.training:
            return x * Annealer.get_trace(self.name)
        else:
            return x