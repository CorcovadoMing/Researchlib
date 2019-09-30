from ..callbacks import *
import torch
from torch.nn.utils import *
from apex import amp
from torch import nn
from ..models import GANModel, VAEModel
from ..utils import *
from ..utils import _register_method

__methods__ = []
register_method = _register_method(__methods__)


def _backup_grad(model):
    for param in model.parameters():
        try:
            param.backup_grad = param.grad.data.clone()
        except:
            if param.requires_grad:
                param.backup_grad = param.data.clone()
                param.backup_grad.zero_()


def _restore_grad(model):
    for param in model.parameters():
        try:
            param.grad = param.backup_grad.data.clone()
        except:
            pass


@register_method
def train_fn(self, train = True, **kwargs):
    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_iteration_begin(**kwargs)

    # Training
    if type(self.model) == GANModel:
        condition = self.model.d_condition or self.model.g_condition

        # Discriminator
        _backup_grad(self.model.generator)
        model = self.model.discriminator
        model_ffn = self.model.forward_d
        loss_ffn = [i.forward_d if isinstance(i, nn.Module) else i for i in self.loss_fn]
        _loss, _norm = _train_minibatch(
            model, model_ffn, loss_ffn, self.optimizer[0], 'unsupervise',
            condition, train, True, False, **kwargs
        )
        for m in kwargs['metrics']:
            m.forward_d([self.model.fake_data_metrics, self.model.real_data])
        _restore_grad(self.model.generator)

        # Record loss
        kwargs['d_loss_history'].append(_loss)

        # Extra
        for i in self.loss_fn:
            i.extra_step(self.model)

        # Generator
        _backup_grad(self.model.discriminator)
        model = self.model.generator
        model_ffn = self.model.forward_g
        loss_ffn = [i.forward_g if isinstance(i, nn.Module) else i for i in self.loss_fn]
        _loss, _norm = _train_minibatch(
            model, model_ffn, loss_ffn, self.optimizer[1], 'unsupervise',
            condition, train, False, False, **kwargs
        )
        for m in kwargs['metrics']:
            m.forward_g([self.model.fake_data_metrics, self.model.real_data])
        _restore_grad(self.model.discriminator)

        # Record loss
        kwargs['g_loss_history'].append(_loss)

    else:
        if type(self.model) == VAEModel:
            learning_type = 'self_supervise'
        else:
            learning_type = 'supervise'
        model = self.model
        model_ffn = self.model.forward
        loss_ffn = [i.forward if isinstance(i, nn.Module) else i for i in self.loss_fn]
        _loss, _norm = _train_minibatch(
            model, model_ffn, loss_ffn, self.optimizer, learning_type, False,
            train, False, False, **kwargs
        )

        # Record loss
        kwargs['loss_history'].append(_loss)

    # Callback: on_iteration_end
    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_iteration_end(**kwargs)
    kwargs['norm'].append(_norm)


def _cal_regularization(_model, **kwargs):
    loss = 0
    regs = get_reg_out(_model)
    for key in kwargs['reg_fn']:
        try:
            weight = kwargs['reg_weights'][key]
        except:
            weight = 1

        reg_args = zip(*regs[key])

        for arg in reg_args:
            reg_loss = (kwargs['reg_fn'][key](*arg)) * weight
            loss += reg_loss.cuda()
    return loss


def _train_minibatch(
    _model, model_ffn, loss_ffn, optim, learning_type, condition, train, retain_graph,
    orthogonal_reg, **kwargs
):
    # Forward
    if condition:
        output = model_ffn(*kwargs['data'], *kwargs['target'])
    else:
        output = model_ffn(*kwargs['data'])

    auxout = get_aux_out(_model)
    if len(auxout) > 0:
        learning_type = 'supervise'
    auxout.append(output)

    # Padding target (local copy, no need to remove), DIRTY TRICK DON'T MODIFIED IT!!
    # This hack just apply on GAN
    while type(_model) == GANModel and len(kwargs['target']) != len(auxout):
        kwargs['target'] = kwargs['target'] + [None]
        loss_ffn.append(loss_ffn[-1])

    # Calulate loss
    optim.zero_grad()
    loss = 0
    auxout = [i.view(i.size(0), -1) for i in auxout]
    if learning_type == 'supervise':
        kwargs['target'] = [i.view(i.size(0), -1) for i in kwargs['target']]
        if kwargs['lam'] is not None:
            kwargs['target_res'] = [i.view(i.size(0), -1) for i in kwargs['target_res']]
        # TODO (Ming): This line should be removed by someway
        if len(kwargs['target']) > len(auxout):
            kwargs['target'] = [kwargs['target']]
        for i in range(len(auxout)):
            if i == 0:
                loss_i = i
                target_i = i
                target_res_i = i
            else:
                loss_i = i if len(loss_ffn) > i else loss_i
                target_i = i if len(kwargs['target']) > i else target_i
                target_res_i = i if kwargs['lam'] is not None and len(
                    kwargs['target_res']
                ) > i else target_res_i

            if kwargs['lam'] is not None:
                loss += kwargs['lam'] * loss_ffn[loss_i](auxout[i], kwargs['target'][target_i])
                loss += (1 - kwargs['lam']
                         ) * loss_ffn[loss_i](auxout[i], kwargs['target_res'][target_res_i])
            else:
                loss += loss_ffn[loss_i](auxout[i], kwargs['target'][target_i].view(-1))
    elif learning_type == 'self_supervise':
        for i in range(len(auxout)):
            loss += loss_ffn[i](auxout[i], *kwargs['data'])
    elif learning_type == 'unsupervise':
        for i in range(len(auxout)):
            loss += loss_ffn[i](auxout[i])

    # Calculate Regularization
    loss += _cal_regularization(_model, **kwargs)

    # Backward
    with amp.scale_loss(loss, optim) as scaled_loss:
        if train:
            scaled_loss.backward(retain_graph = retain_graph)

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_update_begin(**kwargs)

    norm = 0
    if train:
        # Update
        optim.step()
        
        # Calculate norm
        with torch.no_grad():
            for param in _model.parameters():
                try:
                    norm += param.grad.data.norm(2) ** 2
                except:
                    pass
            norm = norm ** 0.5
            norm = norm.cpu()

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_update_end(**kwargs)

    # Apply metrics
    for m in kwargs['metrics']:
        m.forward([auxout[-1]] + [kwargs['target'][-1]])

    return loss.detach().cpu(), norm
