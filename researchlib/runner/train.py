from ..callbacks import *
from tqdm.auto import tqdm
import numpy as np
import torch

class Check:
    def __init__(self):
        pass

def mixup_loss_fn(x, y, y_res, lam, loss_fn):
    return lam * loss_fn(x, y) + (1-lam) * loss_fn(x, y_res)        
            
def train(**kwargs):
    kwargs['check'] = Check()
    kwargs['check'].cutoff = False
    kwargs['model'].train()
    
    loss_history = []
    bar = tqdm(kwargs['train_loader'])
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    for batch_idx, (data, target) in enumerate(bar):
        kwargs['batch_idx'] = batch_idx
        
        if kwargs['augmentor']: data, target = kwargs['augmentor'].on(data, target)
            
        if kwargs['require_long']: target = target.long()
        
        if kwargs['mixup_alpha'] != 0:
            lam = np.random.beta(kwargs['mixup_alpha'], kwargs['mixup_alpha'])
            index = torch.randperm(data.size(0))
            data = lam * data + (1-lam) * data[index]
            target_res = target[index]
            kwargs['check'].lam = lam
            kwargs['mixup_loss_fn'] = mixup_loss_fn
        
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = data.cuda(), target.cuda()
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = target_res.cuda()
        else:
            kwargs['data'], kwargs['target'] = data, target
            if kwargs['mixup_alpha'] != 0: kwargs['target_res'] = target_res
            
        # Callback: on_iteration_begin
        for callback_func in kwargs['callbacks']: callback_func.on_iteration_begin(**kwargs)

        loss_ = train_minibatch_(**kwargs)
        loss_history.append(loss_)
        kwargs['cur_loss'] = loss_
        loss_avg = sum(loss_history)/len(loss_history)
        bar.set_postfix(loss="{:.4f}".format(loss_avg), refresh=False)
        
        # Callback: on_iteration_end
        for callback_func in kwargs['callbacks']: callback_func.on_iteration_end(**kwargs)
        
        # Early stop (for find_lr)
        if kwargs['check'].cutoff: break
    
    # Output metrics
    for m in kwargs['metrics']: m.output()
    
    return loss_history

def train_minibatch_(**kwargs):
    kwargs['optimizer'].zero_grad()
    
    output = kwargs['model'](kwargs['data'])
    
    loss_input = [output, kwargs['target']]
    if kwargs['mixup_alpha'] != 0:
        loss_input.append(kwargs['target_res'])
    
    if not kwargs['keep_x_shape']: loss_input[0] = loss_input[0].contiguous().view(-1, loss_input[0].size(-1))
    if not kwargs['keep_y_shape']:
        for i in range(1, len(loss_input)):
            if len(loss_input[i].shape) > 1:
                loss_input[i] = loss_input[i].contiguous().view(-1, loss_input[i].size(-1))
            else:
                loss_input[i] = loss_input[i].contiguous().view(-1)
                
    if kwargs['require_data']: loss_input.append(kwargs['data'])

    if kwargs['mixup_alpha'] != 0:
        loss_input = loss_input + [kwargs['check'].lam, kwargs['loss_fn']]
        loss = kwargs['mixup_loss_fn'](*loss_input)
    else:
        loss = kwargs['loss_fn'](*loss_input)
        
    loss.backward()
    
    for callback_func in kwargs['callbacks']: callback_func.on_update_begin(**kwargs)
    
    kwargs['optimizer'].step()
    
    for callback_func in kwargs['callbacks']: callback_func.on_update_end(**kwargs)
    
    if type(loss_input[0]) == type(()):
        loss_input[0] = loss_input[0][0]
        loss_input[0] = torch.sqrt((loss_input[0]**2).sum(dim=2, keepdim=True))
    
    # Apply metrics
    for m in kwargs['metrics']: m.forward(loss_input)
        
    return loss.item()
