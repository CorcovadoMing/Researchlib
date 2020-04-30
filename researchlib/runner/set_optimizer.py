from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
import torch_optimizer as extra_optim

'''
AccSGD: https://arxiv.org/abs/1803.05591
AdaBound: https://arxiv.org/abs/1902.09843
AdaMod: https://arxiv.org/abs/1910.12249
DiffGrad: https://arxiv.org/abs/1909.11015
Lamb: https://arxiv.org/abs/1904.00962
NovoGrad: https://arxiv.org/abs/1905.11286
RAdam: https://arxiv.org/abs/1908.03265
SGDW: https://arxiv.org/abs/1608.03983
Yogi: https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
'''

from torch.optim import *
from .optimizer.adafactor import Adafactor
from .optimizer.cocob import Cocob
from .optimizer.lookahead import Lookahead
from .optimizer.nag import NAG
from .optimizer.sm3 import SM3
from .optimizer.bb import BB
from ..utils import _register_method, update_optim
import torchcontrib
from functools import partial, reduce
from .trainable_params_utils import group_parameters, num_list_params
from torchlars import LARS

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def set_optimizer(self, lars=False, info=True):
    opt_mapping = {
        'adam': partial(FusedAdam, adam_w_mode = False, eps = 1e-4),
        'adamw': partial(FusedAdam, eps = 1e-4),
        'lamb': partial(FusedLAMB, eps = 1e-4),
        'novograd': partial(FusedNovoGrad, eps = 1e-4),
        'cocob': Cocob,
        'radam': extra_optim.RAdam,
        'sgd-nomom': partial(FusedSGD, lr = 1e-1),
        'sgd': partial(FusedSGD, lr = 1e-1, momentum = 0.9),
        'accsgd': extra_optim.AccSGD,
        'nesterov': partial(FusedSGD, lr = 1e-2, momentum = 0.9, nesterov = True),
        'nag': partial(NAG, lr = 1e-1),
        'rmsprop': partial(RMSprop, eps = 1e-4),
        'adabound': partial(extra_optim.AdaBound, lr = 1e-3, final_lr = 0.1, eps = 1e-4),
        'adamod': partial(extra_optim.AdaMod, eps = 1e-4),
        'adafactor': partial(Adafactor, lr = 1e-3),
        'sm3': partial(SM3, eps = 1e-4),
        'diffgrad': partial(extra_optim.DiffGrad, eps = 1e-4),
        'sgdw': partial(extra_optim.SGDW, momentum = 0.9),
        'yogi': partial(extra_optim.Yogi, eps = 1e-4),
        'bb': BB,
    }

    loss_params = []
    for i in self.model.optimize_nodes:
        try:
            loss_params += [p for p in self.model.graph(i)[0].parameters() if p.requires_grad]
        except:
            pass

    normal_group, bias_group, no_decay_group, special_group = group_parameters(self.model)
    normal_group_num = num_list_params(normal_group)
    bias_group_num = num_list_params(bias_group)
    no_decay_group_num = num_list_params(no_decay_group)
    special_group_num = num_list_params(special_group)
    
    param_groups = [bias_group, no_decay_group, special_group]
    param_groups_num = [bias_group_num, no_decay_group_num, special_group_num]
    
    if info:
        print('loss_group', reduce(num_list_params, loss_params, 0))
        print('normal_group', normal_group_num)
        print('bias_group', bias_group_num)
        print('special_group', special_group_num)
        print('no_decay_group', no_decay_group_num)

    opt_fn = opt_mapping[self.optimizer_choice]

    self.optimizer = [opt_fn(normal_group + loss_params)]
    for i, j in zip(param_groups_num, param_groups):
        if i != 0:
            self.optimizer.append(opt_fn(j))
    
    for i in range(len(self.optimizer)):
        if self.lookahead:
            self.optimizer[i] = Lookahead(self.optimizer[i])
        if lars:
            self.optimizer[i] = LARS(self.optimizer[i])
