from torch.optim import *
from apex.optimizers import *
from .optimizer.adafactor import Adafactor
from .optimizer.radam import PlainRAdam, RAdam
from .optimizer.adamw import AdamW
from .optimizer.cocob import Cocob
from .optimizer.lookahead import Lookahead
from .optimizer.nag import NAG
from .optimizer.sm3 import SM3
from adabound import AdaBound

from ..utils import _register_method, update_optim
import torchcontrib
from functools import partial, reduce
from .trainable_params_utils import group_parameters, num_list_params
from torchlars import LARS

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def set_optimizer(self, lars=False):
    opt_mapping = {
        'adam': partial(Adam, betas = (0.9, 0.999), eps = 1e-4),
        'adamw': partial(AdamW, betas = (0.9, 0.999), eps = 1e-4),
        'lamb': partial(FusedLAMB, betas = (0.9, 0.999), eps = 1e-4),
        'novograd': FusedNovoGrad,
        'cocob': Cocob,
        'radam-plain': partial(PlainRAdam, betas = (0.9, 0.999), eps = 1e-4),
        'radam': partial(RAdam, betas = (0.9, 0.999), eps = 1e-4),
        'sgd': partial(SGD, lr = 1e-1, momentum = 0.9),
        'nesterov': partial(SGD, lr = 1e-2, momentum = 0.9, nesterov = True),
        'nag': partial(NAG, lr = 1e-1),
        'rmsprop': RMSprop,
        'adabound': partial(AdaBound, lr = 1e-3, final_lr = 0.1),
        'adagrad': Adagrad,
        'adafactor': partial(Adafactor, lr = 1e-3),
        'sm3': partial(SM3, eps = 1e-4),
    }

    loss_params = []
    for i in self.model.optimize_nodes:
        try:
            loss_params += [p for p in self.model.graph(i)[0].parameters() if p.requires_grad]
        except:
            pass

    normal_group, bias_group, no_decay_group = group_parameters(self.model)
    normal_group_num = num_list_params(normal_group)
    bias_group_num = num_list_params(bias_group)
    no_decay_group_num = num_list_params(no_decay_group)
    
    param_groups = [bias_group, no_decay_group]
    param_groups_num = [bias_group_num, no_decay_group_num]
    
    print('loss_group', reduce(num_list_params, loss_params, 0))
    print('normal_group', normal_group_num)
    print('bias_group', bias_group_num)
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
