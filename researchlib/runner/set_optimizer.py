from torch.optim import *
from apex.optimizers import *
from .optimizer.adafactor import Adafactor
from .optimizer.radam import PlainRAdam, RAdam
from .optimizer.adamw import AdamW
from .optimizer.cocob import Cocob
from .optimizer.lookahead import Lookahead
from .optimizer.larc import LARC
from .optimizer.nag import NAG
from adabound import AdaBound
from ..models import GANModel
from ..utils import _register_method, update_optim
import torchcontrib
from functools import partial, reduce
from .trainable_params_utils import is_bias, num_list_params

__methods__ = []
register_method = _register_method(__methods__)


opt_mapping = {
    'adam': partial(Adam, betas = (0.9, 0.999)),
    'adamw': partial(AdamW, betas = (0.9, 0.999)),
    'lamb': partial(FusedLAMB, betas = (0.9, 0.999)),
    'novograd': FusedNovoGrad,
    'cocob': Cocob,
    'radam-plain': partial(PlainRAdam, betas = (0.9, 0.999)),
    'radam': partial(RAdam, betas = (0.9, 0.999)),
    'sgd': partial(SGD, lr = 1e-1, momentum = 0.9),
    'nesterov': partial(SGD, lr = 1e-2, momentum = 0.9, nesterov = True),
    'nag': partial(NAG, lr = 1e-1),
    'rmsprop': RMSprop,
    'adabound': partial(AdaBound, lr = 1e-3, final_lr = 0.1),
    'adagrad': Adagrad,
    'adafactor': partial(Adafactor, lr = 1e-3)
}

@register_method
def set_optimizer(self):
    loss_params = []
    for i in self.loss_fn:
        try:
            loss_params += [p for p in i.parameters() if p.requires_grad]
        except:
            pass
    
    model_weight_params = is_bias(self.model)[False]
    model_bias_params = is_bias(self.model)[True]
    
    print(reduce(num_list_params, loss_params, 0))
    print(num_list_params(model_weight_params))
    print(num_list_params(model_bias_params))
    
    opt_fn = opt_mapping[self.optimizer_choice]
    
    self.optimizer = [opt_fn(model_weight_params + loss_params), 
                      opt_fn(model_bias_params)]
    
    for i in range(len(self.optimizer)):
        if self.lookahead:
            self.optimizer[i] = Lookahead(self.optimizer[i])
        if self.swa:
            self.optimizer[i] = torchcontrib.optim.SWA(self.optimizer[i])
        if self.larc:
            self.optimizer[i] = LARC(self.optimizer[i])