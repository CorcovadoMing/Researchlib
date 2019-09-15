from torch.optim import *
from apex.optimizers import *
from .optimizer.adafactor import Adafactor
from .optimizer.radam import PlainRAdam, RAdam
from .optimizer.adamw import AdamW
from .optimizer.cocob import Cocob
from .optimizer.lookahead import Lookahead
from .optimizer.larc import LARC
from adabound import AdaBound
from ..models import GANModel
from ..utils import _register_method, update_optim
import torchcontrib
from functools import partial

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
            optimizer = Adam(
                list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'lamb':
            optimizer = FusedLAMB(
                list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'novograd':
            optimizer = FusedNovoGrad(list(model.parameters()) + loss_params)
        elif optimizer == 'cocob':
            optimizer = Cocob(list(model.parameters()) + loss_params)
        elif optimizer == 'radam-plain':
            optimizer = PlainRAdam(
                list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'radam':
            optimizer = RAdam(
                list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'adamw':
            optimizer = AdamW(
                list(model.parameters()) + loss_params, betas=(0.9, 0.999))
        elif optimizer == 'sgd':
            optimizer = SGD(
                list(model.parameters()) + loss_params, lr=1e-1, momentum=0.9)
        elif optimizer == 'nesterov':
            optimizer = SGD(
                list(model.parameters()) + loss_params,
                lr=1e-2,
                momentum=0.9,
                nesterov=True)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(list(model.parameters()) + loss_params)
        elif optimizer == 'adabound':
            optimizer = AdaBound(
                list(model.parameters()) + loss_params, lr=1e-3, final_lr=0.1)
        elif optimizer == 'adagrad':
            optimizer = Adagrad(list(model.parameters()) + loss_params)
        elif optimizer == 'adafactor':
            optimizer = Adafactor(
                list(model.parameters()) + loss_params, lr=1e-3)

        if lookahead:
            optimizer = Lookahead(optimizer)
        if swa:
            optimizer = torchcontrib.optim.SWA(optimizer)
        if larc:
            optimizer = LARC(optimizer)
        return optimizer

    _assign_optim_fn = partial(
        _assign_optim, larc=self.larc, swa=self.swa, lookahead=self.lookahead)
    if type(self.model) == GANModel:
        if type(self.optimizer_choice) == list or type(
                self.optimizer_choice) == tuple:
            self.optimizer = [
                _assign_optim_fn(self.model.discriminator,
                                 self.optimizer_choice[1]),
                _assign_optim_fn(self.model.generator, self.optimizer_choice[0])
            ]
        else:
            self.optimizer = [
                _assign_optim_fn(self.model.discriminator,
                                 self.optimizer_choice),
                _assign_optim_fn(self.model.generator, self.optimizer_choice)
            ]
    else:
        self.optimizer = _assign_optim_fn(self.model, self.optimizer_choice)
