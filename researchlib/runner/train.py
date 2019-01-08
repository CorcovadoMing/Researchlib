from ..callbacks import *
from tqdm.auto import tqdm

def train(**kwargs):
    kwargs['model'].train()
    loss_history = []
    for batch_idx, (data, target) in enumerate(tqdm(kwargs['train_loader'])):
        kwargs['batch_idx'] = batch_idx
        
        if kwargs['require_long']:
            target = target.long()
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = data.cuda(), target.cuda()
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_begin(**kwargs)

        loss_ = train_minibatch(**kwargs)
        loss_history.append(loss_)
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_end(**kwargs)
    
    return loss_history

def train_minibatch(**kwargs):
    kwargs['optimizer'].zero_grad()
    output = kwargs['model'](kwargs['data'])

    loss_input = [output, kwargs['target']]
    
    if kwargs['require_data']:
        loss_input.append(kwargs['data'])

    if not kwargs['keep_shape']:
        loss_input[0].view(loss_input[0].size(0), -1)
        loss_input[1].view(-1,)
    
    loss = kwargs['loss_fn'](*loss_input)

    loss.backward()
    kwargs['optimizer'].step()
    return loss.item()
