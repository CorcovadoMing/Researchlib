from ..callbacks import *
import numpy as np
import torch
from torch.nn.utils import *
from ..utils import *
from apex import amp
import torchtext
from torch import nn
from ..models import GANModel

def train_fn(**kwargs):
    # Callback: on_iteration_begin
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)

    # Training
    if type(kwargs['model']) == GANModel:
        condition = kwargs['model'].d_condition or kwargs['model'].g_condition

        # Discriminator
        model = kwargs['model'].discriminator
        model_ffn = kwargs['model'].forward_d
        loss_ffn = [i.forward_d if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
        loss_ = _train_minibatch(model, model_ffn, loss_ffn, kwargs['optimizer'][0], True, condition, **kwargs)

        # Record loss
        kwargs['d_loss_history'].append(loss_)
        d_loss_avg = sum(kwargs['d_loss_history'])/len(kwargs['d_loss_history'])

        # Extra
        for i in kwargs['loss_fn']:
            i.extra_step(kwargs['model'])

        # Generator
        model_ffn = kwargs['model'].forward_g
        loss_ffn = [i.forward_g if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
        loss_ = _train_minibatch(model, model_ffn, loss_ffn, kwargs['optimizer'][1], True, condition, **kwargs)

        # Record loss
        kwargs['g_loss_history'].append(loss_)
        kwargs['cur_loss'] = loss_
        g_loss_avg = sum(kwargs['g_loss_history'])/len(kwargs['g_loss_history'])
        if kwargs['bar']: kwargs['bar'].set_postfix(d_loss="{:.4f}".format(d_loss_avg), g_loss="{:.4f}".format(g_loss_avg), refresh=False)

    else:
        model = kwargs['model']
        model_ffn = kwargs['model'].forward
        loss_ffn = [i.forward if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
        loss_ = _train_minibatch(model, model_ffn, loss_ffn, kwargs['optimizer'], False, False, **kwargs)

        # Record loss
        kwargs['loss_history'].append(loss_)
        kwargs['cur_loss'] = loss_
        loss_avg = sum(kwargs['loss_history'])/len(kwargs['loss_history'])
        if kwargs['bar']: kwargs['bar'].set_postfix(loss="{:.4f}".format(loss_avg), refresh=False)

    # Callback: on_iteration_end
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_end(**kwargs)
    
    
def cal_regularization(_model, **kwargs):
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

def _train_minibatch(_model, model_ffn, loss_ffn, optim, unsupervised, condition, **kwargs):
    # Reset optimizer
    optim.zero_grad()
    
    # Forward
    if condition:
        output = model_ffn(*kwargs['data'], *kwargs['target'])
    else:
        output = model_ffn(*kwargs['data'])
    
    auxout = get_aux_out(_model)
    if len(auxout) > 0:
        unsupervised = False
    auxout.append(output)
    
    # Padding target (local copy, no need to remove), DIRTY TRICK DON'T MODIFIED IT!!
    while len(kwargs['target']) != len(auxout):
        kwargs['target'] = kwargs['target'] + [None]
        kwargs['keep_y_shape'] = kwargs['keep_y_shape'] + [True]
        kwargs['keep_x_shape'] = kwargs['keep_x_shape'] + [True]
        loss_ffn.append(loss_ffn[-1])

    # Calulate loss
    loss = 0
    auxout = [i if j else i.view(i.size(0), -1) for i, j in zip(auxout, kwargs['keep_x_shape'])]
    if not unsupervised:
        kwargs['target'] = [i if j else i.view(i.size(0), -1) for i, j in zip(kwargs['target'], kwargs['keep_y_shape'])]
        for i in range(len(auxout)):
            if kwargs['mixup_alpha'] != 0:
                if not kwargs['keep_y_shape'][i]:
                    kwargs['target_res'][i] = kwargs['target_res'][i].view(kwargs['target_res'][i].size(0), -1)
                loss += kwargs['mixup_loss_fn'](loss_ffn[i], auxout[i], kwargs['target'][i], kwargs['target_res'][i], kwargs['check'].lam)
            else:
                loss += loss_ffn[i](auxout[i], kwargs['target'][i])
    else:
        for i in range(len(auxout)):
            if kwargs['mixup_alpha'] != 0:
                loss += kwargs['mixup_loss_fn'](loss_ffn[i], auxout[i], kwargs['check'].lam)
            else:
                loss += loss_ffn[i](auxout[i])
    
    # Calculate Regularization
    loss += cal_regularization(_model, **kwargs)

    # Backward
    with amp.scale_loss(loss, optim) as scaled_loss:
        scaled_loss.backward()
        del scaled_loss
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_begin(**kwargs)
    
    # Update
    optim.step()
    optim.zero_grad()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_end(**kwargs)
    
    # Apply metrics
    for m in kwargs['metrics']: m.forward([auxout[-1]] + [kwargs['target'][-1]])
    
    record = loss.detach().cpu()
    return record
