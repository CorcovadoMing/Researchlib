from ..callbacks import *
from tqdm.auto import tqdm

class Check:
    def __init__(self):
        pass

def advtrain(**kwargs):
    kwargs['check'] = Check()
    kwargs['check'].cutoff = False
    kwargs['model'].train()
    loss_history = []
    bar = tqdm(kwargs['train_loader'])
    for batch_idx, (data, target) in enumerate(bar):
        kwargs['batch_idx'] = batch_idx
        
        if kwargs['require_long']:
            target = target.long()
        if kwargs['is_cuda']:
            kwargs['data'], kwargs['target'] = data.cuda(), target.cuda()
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_begin(**kwargs)

        loss_ = advtrain_minibatch_(**kwargs)
        loss_history.append(loss_)
        kwargs['cur_loss'] = loss_
        bar.set_postfix(loss="{:.4f}".format(loss_), refresh=False)
        
        for callback_func in kwargs['callbacks']:
            callback_func.on_iteration_end(**kwargs)
        
        if kwargs['check'].cutoff:
            break
    
    return loss_history

def advtrain_minibatch_(**kwargs):
    kwargs['optimizer'].zero_grad()
    output = kwargs['model'](kwargs['data'])

    loss_input = [output, kwargs['target']]
    
    if kwargs['require_data']:
        loss_input.append(kwargs['data'])

    if not kwargs['keep_shape']:
        loss_input[0] = loss_input[0].view(-1, loss_input[0].size(-1))
        loss_input[1] = loss_input[1].view(-1,)
        
    loss = kwargs['loss_fn'](*loss_input)

    loss.backward()
    kwargs['optimizer'].step()
    return loss.item()
