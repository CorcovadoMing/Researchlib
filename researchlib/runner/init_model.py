import torch.nn.init as init
from ..utils import _register_method
from torch import nn

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def init_model(self, init_distribution='xavier_normal', verbose=False):
    def _init(m):
        if type(m) == nn.ModuleList or 'researchlib.layers.block' in str(type(m)) or 'researchlib.models' in str(type(m)):
            pass
        else:
            if verbose:
                print('Initialize to ' + str(init_distribution) + ' :', m)
            for i in m.parameters():
                try:
                    if init_distribution == 'xavier_normal':
                        init.xavier_normal_(i)
                    elif init_distribution == 'orthogonal':
                        init.orthogonal_(i)
                except:
                    init.uniform_(i)
    self.model.apply(_init)
                
