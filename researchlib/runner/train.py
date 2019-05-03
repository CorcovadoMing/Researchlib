from ..callbacks import *
from .history import *
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.nn.utils import *
from ..utils import *
from apex import amp
import torchtext
from torch import nn
from ..models import *

class Check:
    def __init__(self):
        pass

def mixup_loss_fn(loss_fn, x, y, y_res, lam):
    return lam * loss_fn(x, y) + (1 - lam) * loss_fn(x, y_res)


def train_fn(**kwargs):
    kwargs['check'] = Check()
    kwargs['check'].cutoff = False
    kwargs['model'].train()
    
    loss_history = []
    g_loss_history = []
    d_loss_history = []
    matrix_records = History()
    
    bar = tqdm(kwargs['train_loader'], leave=False)
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    for batch_idx, data_pack in enumerate(bar):
        kwargs['batch_idx'] = batch_idx
        
        # Load dataset
        if type(kwargs['train_loader']) == torchtext.data.iterator.BucketIterator:
            data, target = data_pack.text, data_pack.label
        elif type(data_pack[0]) == type({}):
            data, target = data_pack[0]['data'], data_pack[0]['label']
        else:
            data, target = data_pack[0], data_pack[1:]
        
        if type(data) != type([]) and type(data) != type(()): data = [data]
        if type(target) != type([]) and type(target) != type(()): target = [target]
        
        target = [i.long() if j else i for i, j in zip(target, kwargs['require_long'])]
        
        # On the fly augmentation
        if kwargs['augmentor']: data, target = kwargs['augmentor'].on(data, target)
        
        # Mixup
        if kwargs['mixup_alpha'] != 0:
            lam = np.random.beta(kwargs['mixup_alpha'], kwargs['mixup_alpha'])
            index = torch.randperm(data[0].size(0))
            data[0] = lam * data[0] + (1-lam) * data[0][index]
            target_res = [i[index] for i in target]
            kwargs['check'].lam = lam
            kwargs['mixup_loss_fn'] = mixup_loss_fn
        
        # Move to GPU
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = [i.cuda() for i in data], [i.cuda() for i in target]
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = [i.cuda() for i in target_res]
        else:
            kwargs['data'], kwargs['target'] = data, target
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = target_res
            
        # Callback: on_iteration_begin
        for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)

        # Training
        if type(kwargs['model']) == GANModel:
            condition = kwargs['model'].d_condition or kwargs['model'].g_condition
        
            # Discriminator
            model = kwargs['model'].discriminator
            model_ffn = kwargs['model'].forward_d
            loss_ffn = [i.forward_d if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
            loss_ = train_minibatch_(model, model_ffn, loss_ffn, kwargs['optimizer'][0], True, condition, **kwargs)
            
            # Record loss
            d_loss_history.append(loss_)
            d_loss_avg = sum(d_loss_history)/len(d_loss_history)
            
            # Extra
            for i in kwargs['loss_fn']:
                i.extra_step(kwargs['model'])
            
            # Generator
            #model = kwargs['model'].generator
            model_ffn = kwargs['model'].forward_g
            loss_ffn = [i.forward_g if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
            loss_ = train_minibatch_(model, model_ffn, loss_ffn, kwargs['optimizer'][1], True, condition, **kwargs)
            
            # Record loss
            g_loss_history.append(loss_)
            kwargs['cur_loss'] = loss_
            g_loss_avg = sum(g_loss_history)/len(g_loss_history)
            bar.set_postfix(d_loss="{:.4f}".format(d_loss_avg), g_loss="{:.4f}".format(g_loss_avg), refresh=False)
            
        else:
            model = kwargs['model']
            model_ffn = kwargs['model'].forward
            loss_ffn = [i.forward if isinstance(i, nn.Module) else i for i in kwargs['loss_fn']]
            loss_ = train_minibatch_(model, model_ffn, loss_ffn, kwargs['optimizer'], False, False, **kwargs)
        
            # Record loss
            loss_history.append(loss_)
            kwargs['cur_loss'] = loss_
            loss_avg = sum(loss_history)/len(loss_history)
            bar.set_postfix(loss="{:.4f}".format(loss_avg), refresh=False)
        
        # Callback: on_iteration_end
        for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_end(**kwargs)
        
        # Early stop (for find_lr)
        if kwargs['check'].cutoff: break
    
    # Output metrics
    for m in kwargs['metrics']: matrix_records.add(m.output(), prefix='train')
    
    if type(kwargs['model']) == GANModel:
        return {'d_loss': sum(d_loss_history)/len(d_loss_history), 'g_loss': sum(g_loss_history)/len(g_loss_history)}, matrix_records
    else:
        return {'loss': sum(loss_history)/len(loss_history)}, matrix_records
    
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
            arg = [i.cpu() for i in arg]
            reg_loss = (kwargs['reg_fn'][key](*arg)) * weight
            loss += reg_loss.cuda()
    return loss

def train_minibatch_(_model, model_ffn, loss_ffn, optim, unsupervised, condition, **kwargs):
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
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_begin(**kwargs)
    
    # Update
    optim.step()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_end(**kwargs)
    
    # Apply metrics
    for m in kwargs['metrics']: m.forward([auxout[-1]] + [kwargs['target'][-1]])
    return loss.item()
