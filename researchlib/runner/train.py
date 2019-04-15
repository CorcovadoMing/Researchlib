from ..callbacks import *
from .history import *
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.nn.utils import *
from ..utils import *
from apex import amp

class Check:
    def __init__(self):
        pass

def mixup_loss_fn(loss_fn, x, y, y_res, lam):
    return lam * loss_fn(x, y) + (1 - lam) * loss_fn(x, y_res)


def train(**kwargs):
    kwargs['check'] = Check()
    kwargs['check'].cutoff = False
    kwargs['model'].train()
    
    loss_history = []
    matrix_records = History()
    
    bar = tqdm(kwargs['train_loader'], leave=False)
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    for batch_idx, data_pack in enumerate(bar):
        if type(data_pack[0]) == type({}):
            data, target = data_pack[0]['data'], data_pack[0]['label']
        else:
            data, target = data_pack[0], data_pack[1:]
        
        kwargs['batch_idx'] = batch_idx
        
        if kwargs['augmentor']: data, target = kwargs['augmentor'].on(data, target)
        
        if type(data) != type([]) and type(data) != type(()): data = [data]
        if type(target) != type([]) and type(target) != type(()): target = [target]
        
        target = [i.long() if j else i for i, j in zip(target, kwargs['require_long'])]
        
        if kwargs['mixup_alpha'] != 0:
            lam = np.random.beta(kwargs['mixup_alpha'], kwargs['mixup_alpha'])
            index = torch.randperm(data[0].size(0))
            data[0] = lam * data[0] + (1-lam) * data[0][index]
            target_res = [i[index] for i in target]
            kwargs['check'].lam = lam
            kwargs['mixup_loss_fn'] = mixup_loss_fn
        
        
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = [i.cuda() for i in data], [i.cuda() for i in target]
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = [i.cuda() for i in target_res]
        else:
            kwargs['data'], kwargs['target'] = data, target
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = target_res
            
        # Callback: on_iteration_begin
        for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)

        loss_ = train_minibatch_(**kwargs)
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
    
    return loss_history, matrix_records
    

def train_minibatch_(**kwargs):
    kwargs['optimizer'].zero_grad()
    
    output = kwargs['model'](*kwargs['data'])
    auxout = get_aux_out(kwargs['model'])
    auxout.append(output)
    
    auxout = [i if j else i.contiguous().view(-1, *tuple(i.shape)[1:]) for i, j in zip(auxout, kwargs['keep_x_shape'])]
    kwargs['target'] = [i.squeeze() if j else i.contiguous().view(-1, *tuple(i.shape)[1:]) for i, j in zip(kwargs['target'], kwargs['keep_y_shape'])]
    if kwargs['mixup_alpha'] != 0:
        kwargs['target_res'] = [i.squeeze() if j else i.contiguous().view(-1, *tuple(i.shape)[1:]) for i, j in zip(kwargs['target_res'], kwargs['keep_y_shape'])]

    loss = 0
    if kwargs['mixup_alpha'] != 0:
        for i in range(len(auxout)):
            loss += kwargs['mixup_loss_fn'](kwargs['loss_fn'][i], auxout[i], kwargs['target'][i], kwargs['target_res'][i], kwargs['check'].lam)
    else:
        for i in range(len(auxout)):
            loss += kwargs['loss_fn'][i](auxout[i], kwargs['target'][i])
        
    # Reg
    regs = get_reg_out(kwargs['model'])
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

    with amp.scale_loss(loss, kwargs['optimizer']) as scaled_loss:
        scaled_loss.backward()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_begin(**kwargs)
    
    kwargs['optimizer'].step()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_end(**kwargs)
    
    # Apply metrics
    for m in kwargs['metrics']: m.forward([auxout[-1]] + [kwargs['target'][-1]])
        
    return loss.item()
