from torch.optim import *
from .optimizer.adafactor import Adafactor
from .optimizer.radam import PlainRAdam, RAdam
from .optimizer.adamw import AdamW
from .optimizer.cocob import Cocob
from .optimizer.lookahead import Lookahead
from .optimizer.larc import LARC
from adabound import AdaBound
from ..models import GANModel
from ..utils import _register_method
import torchcontrib

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def set_optimizer(self):
    def _assign_optim(model, optimizer, larc, swa, lookahead):
        # if there are learnable loss parameters
        loss_params = []
        for i in self.loss_fn:
            try:
                loss_params += i.parameters()
            except:
                pass

        if optimizer == 'adam':
            optimizer = Adam(list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'cocob':
            optimizer = Cocob(list(model.parameters()) + loss_params)
        elif optimizer == 'radam-plain':
            optimizer = PlainRAdam(list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'radam':
            optimizer = RAdam(list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'adamw':
            optimizer = AdamW(list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'adam-gan':
            optimizer = Adam(list(model.parameters()) + loss_params, betas=(0., 0.999))
        elif optimizer == 'sgd':
            optimizer = SGD(list(model.parameters()) + loss_params, lr=1e-1, momentum=0.9)
        elif optimizer == 'nesterov':
            optimizer = SGD(list(model.parameters()) + loss_params, lr=1e-2, momentum=0.9, nesterov=True)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(list(model.parameters()) + loss_params)
        elif optimizer == 'adabound':
            optimizer = AdaBound(list(model.parameters()) + loss_params, lr=1e-3, final_lr=0.1)
        elif optimizer == 'adagrad':
            optimizer = Adagrad(list(model.parameters()) + loss_params)
        elif optimizer == 'adafactor':
            optimizer = Adafactor(list(model.parameters()) + loss_params, lr=1e-3)
            
        if larc:
            optimizer = LARC(optimizer)
        if lookahead:
            optimizer = Lookahead(optimizer)
        if swa:
            optimizer = torchcontrib.optim.SWA(optimizer)
        return optimizer

    if type(self.model) == GANModel:
        if type(self.optimizer_choice) == list or type(self.optimizer_choice) == tuple:
            self.optimizer = [
                _assign_optim(self.model.discriminator, self.optimizer_choice[1], 
                              self.larc, self.swa, self.lookahead),
                _assign_optim(self.model.generator, self.optimizer_choice[0], 
                              self.larc, self.swa, self.lookahead)
            ]
        else:
            self.optimizer = [
                _assign_optim(self.model.discriminator, self.optimizer_choice, 
                              self.larc, self.swa,self.lookahead),
                _assign_optim(self.model.generator, self.optimizer_choice,
                              self.larc, self.swa, self.lookahead)
            ]
    else:
        self.optimizer = _assign_optim(self.model, self.optimizer_choice,
                                       self.larc, self.swa, self.lookahead)