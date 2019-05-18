import torch.nn.init as init
from ..utils import _register_method, _is_container

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def init_model(self, init_distribution='xavier_normal', module_list=[], verbose=False):                
    def _is_init_module(m, module_list):
        if _is_container(m):
            return False
        if len(module_list):
            if type(m) in module_list:
                return True
            else:
                return False
        else:
            return True

    def _init(m):
        if _is_init_module(m, module_list):
            for p in m.parameters():
                if p.dim() > 1:
                    if init_distribution == 'xavier_normal':
                        init.xavier_normal_(p.data)
                    elif init_distribution == 'orthogonal':
                        init.orthogonal_(p.data)
                    if verbose:
                        print('Initialize to ' + str(init_distribution) + ':', m)
                else:
                    init.normal_(p.data)
    self.model.apply(_init)
