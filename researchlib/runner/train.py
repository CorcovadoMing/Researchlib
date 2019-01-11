from ..callbacks import *
from tqdm.auto import tqdm
import torch

class Check:
    def __init__(self):
        pass

def train(**kwargs):
    kwargs['check'] = Check()
    kwargs['check'].cutoff = False
    kwargs['model'].train()
    loss_history = []
    bar = tqdm(kwargs['train_loader'])
    for batch_idx, (data, target) in enumerate(bar):
        kwargs['batch_idx'] = batch_idx
        if kwargs['augmentor']:
            data, target = kwargs['augmentor'].on(data, target)
            
        if kwargs['require_long']:
            target = target.long()
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = data.cuda(), target.cuda()
        else:
            kwargs['data'], kwargs['target'] = data, target
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_begin(**kwargs)

        loss_ = train_minibatch_(**kwargs)
        loss_history.append(loss_)
        kwargs['cur_loss'] = loss_
        bar.set_postfix(loss="{:.4f}".format(loss_), refresh=False)
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_end(**kwargs)
        
        if kwargs['check'].cutoff:
            break
    
    return loss_history

def train_minibatch_(**kwargs):
    kwargs['optimizer'].zero_grad()
    
    output = kwargs['model'](kwargs['data'])
    
    loss_input = [output, kwargs['target']]
    
    if kwargs['require_data']:
        loss_input.append(kwargs['data'])
    
    if not kwargs['keep_x_shape']:
        loss_input[0] = loss_input[0].contiguous().view(-1, loss_input[0].size(-1))
    if not kwargs['keep_y_shape']:
        loss_input[1] = loss_input[1].contiguous().view(-1)
    else:
        loss_input[1] = loss_input[1].contiguous().view(-1, loss_input[1].size(-1))

    loss = kwargs['loss_fn'](*loss_input)

    loss.backward()
    
    kwargs['optimizer'].step()
    
    return loss.item()
