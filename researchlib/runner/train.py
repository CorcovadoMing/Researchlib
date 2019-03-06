from ..callbacks import *
from .history import *
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.nn.utils import *
from ..utils import *

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
    
    for batch_idx, (data, target) in enumerate(bar):
        kwargs['batch_idx'] = batch_idx
        
        if kwargs['augmentor']: data, target = kwargs['augmentor'].on(data, target)
            
        if kwargs['require_long']: target = target.long()
        
        
        if type(data) != type([]):
            data = [data]
        
        if type(target) != type([]):
            target = [target]
        
        
        if kwargs['mixup_alpha'] != 0:
            lam = np.random.beta(kwargs['mixup_alpha'], kwargs['mixup_alpha'])
            index = torch.randperm(data[0].size(0))
            data[0] = lam * data[0] + (1-lam) * data[0][index]
            target_res = target[0][index]
            target_res = [target_res]
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
    
    loss_input = [*kwargs['target']]
    if kwargs['mixup_alpha'] != 0:
        loss_input.append(*kwargs['target_res'])
    
    if not kwargs['keep_x_shape']: auxout[-1] = auxout[-1].contiguous().view(-1, auxout[-1].size(-1))
    if not kwargs['keep_y_shape']:
        for i in range(1, len(loss_input)):
            if len(loss_input[i].shape) > 1:
                loss_input[i] = loss_input[i].contiguous().view(-1, loss_input[i].size(-1))
            else:
                loss_input[i] = loss_input[i].contiguous().view(-1)
                
    if kwargs['require_data']: loss_input.append(kwargs['data'])

    loss = torch.zeros(1).cuda()
    if kwargs['mixup_alpha'] != 0:
        loss_input = loss_input + [kwargs['check'].lam]
        for i in range(len(auxout)):
            loss += kwargs['mixup_loss_fn'](kwargs['loss_fn'][i], auxout[i], *loss_input)
    else:
        for i in range(len(auxout)):
            loss += kwargs['loss_fn'][i](auxout[i], *loss_input)
        
    loss.backward()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_begin(**kwargs)
    
    clip_grad_norm_(kwargs['model'].parameters(), 5.)
    kwargs['optimizer'].step()
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_update_end(**kwargs)
    
    if type(auxout[-1]) == type(()):
        auxout[-1] = auxout[-1][0]
        auxout[-1] = torch.sqrt((auxout[-1]**2).sum(dim=2, keepdim=True))
    
    # Apply metrics
    for m in kwargs['metrics']: m.forward([auxout[-1]] + loss_input)
        
    return loss.item()
