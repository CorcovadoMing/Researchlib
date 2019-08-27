from ..callbacks import *
import numpy as np
import torch
from torch.nn.utils import *
from ..utils import *
from apex import amp
import torchtext
from torch import nn
from ..models import GANModel, VAEModel
from ..utils import _register_method
import copy
import functools
import math

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
def train_fn(self, train=True, **kwargs):
    ema = self.ema > 0 and self.epoch > self.ema_start
    # Callback: on_iteration_begin
    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_iteration_begin(**kwargs)

    # Training
    if type(self.model) == GANModel:
        # For metrics
        self.model.ema = ema

        condition = self.model.d_condition or self.model.g_condition

        if ema:
            for p in self.model.generator.parameters():
                if not hasattr(p, 'ema'):
                    p.ema = p.data.clone()

        # Discriminator
        _backup_grad(self.model.generator)
        model = self.model.discriminator
        model_ffn = self.model.forward_d
        loss_ffn = [
            i.forward_d if isinstance(i, nn.Module) else i for i in self.loss_fn
        ]
        _loss, _norm = _train_minibatch(model, model_ffn, loss_ffn,
                                        self.optimizer[0], self.scheduler,
                                        'unsupervise', condition, train, True,
                                        False, self.ema, **kwargs)
        for m in kwargs['metrics']:
            m.forward_d([self.model.fake_data_metrics, self.model.real_data])
        _restore_grad(self.model.generator)

        # Record loss
        kwargs['d_loss_history'].append(_loss)
        d_loss_avg = sum(kwargs['d_loss_history']) / len(
            kwargs['d_loss_history'])

        # Extra
        for i in self.loss_fn:
            i.extra_step(self.model)

        # Generator
        _backup_grad(self.model.discriminator)
        model = self.model.generator
        model_ffn = functools.partial(self.model.forward_g)
        loss_ffn = [
            i.forward_g if isinstance(i, nn.Module) else i for i in self.loss_fn
        ]
        _loss, _norm = _train_minibatch(model, model_ffn, loss_ffn,
                                        self.optimizer[1], self.scheduler,
                                        'unsupervise', condition, train, False,
                                        False, self.ema, **kwargs)
        for m in kwargs['metrics']:
            m.forward_g([self.model.fake_data_metrics, self.model.real_data])
        _restore_grad(self.model.discriminator)

        # Record loss
        kwargs['g_loss_history'].append(_loss)
        kwargs['cur_loss'] = _loss
        g_loss_avg = sum(kwargs['g_loss_history']) / len(
            kwargs['g_loss_history'])
        if kwargs['bar']:
            kwargs['bar'].set_postfix(
                d_loss="{:.4f}".format(d_loss_avg),
                g_loss="{:.4f}".format(g_loss_avg),
                refresh=False)

    else:
        if ema:
            for p in self.model.parameters():
                if not hasattr(p, 'ema'):
                    p.ema = p.data.clone()

        if type(self.model) == VAEModel:
            learning_type = 'self_supervise'
        else:
            learning_type = 'supervise'
        model = self.model
        model_ffn = self.model.forward
        loss_ffn = [
            i.forward if isinstance(i, nn.Module) else i for i in self.loss_fn
        ]
        _loss, _norm = _train_minibatch(model, model_ffn, loss_ffn,
                                        self.optimizer, self.scheduler,
                                        learning_type, False, train, False,
                                        False, self.ema, **kwargs)

        # Record loss
        kwargs['loss_history'].append(_loss)
        kwargs['cur_loss'] = _loss
        loss_avg = sum(kwargs['loss_history']) / len(kwargs['loss_history'])
        if kwargs['bar']:
            kwargs['bar'].set_postfix(
                loss="{:.4f}".format(loss_avg), refresh=False)

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


def _train_minibatch(_model, model_ffn, loss_ffn, optim, scheduler,
                     learning_type, condition, train, retain_graph,
                     orthogonal_reg, ema, **kwargs):
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
    loss = 0
    auxout = [i.view(i.size(0), -1) for i in auxout]
    if learning_type == 'supervise':
        kwargs['target'] = [i.view(i.size(0), -1) for i in kwargs['target']]
        if kwargs['mixup_alpha'] != 0:
            kwargs['target_res'] = [
                i.view(i.size(0), -1) for i in kwargs['target_res']
            ]
        if len(kwargs['target']) > len(auxout):
            kwargs['target'] = [kwargs['target']]
        for i in range(len(auxout)):
            if kwargs['mixup_alpha'] != 0:
                loss += kwargs['mixup_loss_fn'](loss_ffn[i], auxout[i],
                                                kwargs['target'][i],
                                                kwargs['target_res'][i],
                                                kwargs['check'].lam)
            else:
                loss += loss_ffn[i](auxout[i], kwargs['target'][i])
    elif learning_type == 'self_supervise':
        for i in range(len(auxout)):
            if kwargs['mixup_alpha'] != 0:
                loss += kwargs['mixup_loss_fn'](loss_ffn[i], auxout[i],
                                                kwargs['check'].lam,
                                                *kwargs['data'])
            else:
                loss += loss_ffn[i](auxout[i], *kwargs['data'])
    elif learning_type == 'unsupervise':
        for i in range(len(auxout)):
            if kwargs['mixup_alpha'] != 0:
                loss += kwargs['mixup_loss_fn'](loss_ffn[i], auxout[i],
                                                kwargs['check'].lam)
            else:
                loss += loss_ffn[i](auxout[i])

    # Calculate Regularization
    loss += _cal_regularization(_model, **kwargs)

    # Backward
    with amp.scale_loss(loss, optim) as scaled_loss:
        if train: 
            scaled_loss.backward(retain_graph=retain_graph)
#     if train:
#         loss.backward(retain_graph=retain_graph)

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_update_begin(**kwargs)

    # Update
    norm = 0
    if train:
        with torch.no_grad():
            if orthogonal_reg:
                for param in _model.parameters():
                    # Only apply this to parameters with at least 2 axes, and not in the blacklist
                    if len(param.shape) < 2:
                        continue
                    w = param.view(param.shape[0], -1)
                    grad = (2 * torch.mm(
                        torch.mm(w, w.t()) *
                        (1. - torch.eye(w.shape[0], device=w.device)), w))
                    param.grad.data += 1e-4 * grad.view(param.shape)

            for param in _model.parameters():
                try:
                    norm += param.grad.data.norm(2)**2
                except:
                    pass
            norm = norm**0.5
            norm = norm.detach().cpu()

        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad()

        with torch.no_grad():
            if ema > 0:
                for param in _model.parameters():
                    if hasattr(param, 'ema'):
                        param.ema.data = (ema * param.ema.data) + (
                            (1 - ema) * param.data)

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_update_end(**kwargs)

    # Apply metrics
    for m in kwargs['metrics']:
        m.forward([auxout[-1]] + [kwargs['target'][-1]])

    record = loss.detach().cpu()
    return record, norm
