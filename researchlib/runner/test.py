import torch
import torch.nn.functional as F
from .history import *
from ..utils import *

def test(**kwargs):
    '''
        kwargs: model, test_loader, loss_fn, is_cuda, require_long, keep_x_shape, keep_y_shape, metrics
    '''
    
    kwargs['model'].eval()
    test_loss = 0
    matrix_records = History()
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    with torch.no_grad():
        for batch_idx, data_pack in enumerate(kwargs['test_loader']):
            if type(data_pack[0]) == type({}):
                data, target = data_pack[0]['data'], data_pack[0]['label']
            else:
                data, target = data_pack[0], data_pack[1:]
                
            kwargs['batch_idx'] = batch_idx

            if type(target) != type([]) and type(target) != type(()): target = [target]
        
            target = [i.long() if j else i for i, j in zip(target, kwargs['require_long'])]
            
            if kwargs['is_cuda']: data, target = data.cuda(), [i.cuda() for i in target]
            
            kwargs['data'], kwargs['target'] = data, target
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)
            
            if type(kwargs['data']) == type([]):
                output = kwargs['model'](*kwargs['data'])
            else:
                output = kwargs['model'](kwargs['data'])
            
            auxout = get_aux_out(kwargs['model'])
            auxout.append(output)
            kwargs['auxout'] = auxout
            
            auxout = [i if j else i.contiguous().view(-1, *tuple(i.shape)[1:]) for i, j in zip(auxout, kwargs['keep_x_shape'])]
            kwargs['target'] = [i.squeeze() if j else i.contiguous().view(-1, *tuple(i.shape)[1:]) for i, j in zip(kwargs['target'], kwargs['keep_y_shape'])]
    
            for i in range(len(auxout)):
                test_loss += kwargs['loss_fn'][i](auxout[i], kwargs['target'][i]).item()
            
            for m in kwargs['metrics']: m.forward([auxout[-1]] + [kwargs['target'][-1]])
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_end(**kwargs)

    test_loss /= (kwargs['batch_idx'] + 1)
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_validation_end(**kwargs)

    # Output metrics
    for m in kwargs['metrics']: matrix_records.add(m.output(), prefix='val')    
    

    return test_loss, matrix_records