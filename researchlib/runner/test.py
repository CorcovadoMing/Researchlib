import torch
import torch.nn.functional as F
from .history import *
from ..utils import *

def test(**kwargs):
    '''
        kwargs: model, test_loader, loss_fn, is_cuda, require_long, require_data, keep_x_shape, keep_y_shape, metrics
    '''
    
    kwargs['model'].eval()
    test_loss = 0
    matrix_records = History()
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    with torch.no_grad():
        for batch_idx, data_pack in enumerate(kwargs['test_loader']):
            data = data_pack[0]
            target = data_pack[1:]
        
            if kwargs['require_long']: target = [target[0].float(), target[-1].long()]
            if kwargs['is_cuda']: data, target = data.cuda(), [i.cuda() for i in target]
            
            kwargs['data'], kwargs['target'] = data, target
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)
            
            if type(kwargs['data']) == type([]):
                output = kwargs['model'](*kwargs['data'])
            else:
                output = kwargs['model'](kwargs['data'])
            
            auxout = get_aux_out(kwargs['model'])
            auxout.append(output)
            
            loss_input = [kwargs['target']]
            
            if not kwargs['keep_x_shape']: auxout[-1] = auxout[-1].contiguous().view(-1, auxout[-1].size(-1))
            if not kwargs['keep_y_shape']:
                for i in range(len(loss_input)):
                    for j in range(len(loss_input[i])):
                        if len(loss_input[i][j].shape) > 1:
                            loss_input[i][j] = loss_input[i][j].contiguous().view(-1, loss_input[i][j].size(-1))
                        else:
                            loss_input[i][j] = loss_input[i][j].contiguous().view(-1)

            kwargs['target'] = loss_input[0]
            #if kwargs['require_data']: loss_input.append(data)
            
            for i in range(len(auxout)):
                test_loss += kwargs['loss_fn'][i](auxout[i], kwargs['target'][i]).item()
            
            # Capsule, TODO: refine the part
            if type(auxout[-1]) == type(()):
                auxout[-1] = auxout[-1][0]
                auxout[-1] = torch.sqrt((auxout[-1]**2).sum(dim=2, keepdim=True))
            
            for m in kwargs['metrics']: m.forward([auxout[-1]] + [kwargs['target'][-1]])
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_end(**kwargs)

    # Output metrics
    for m in kwargs['metrics']: matrix_records.add(m.output(), prefix='val')    
    
    test_loss /= len(kwargs['test_loader'])
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_validation_end(**kwargs)
    
    return test_loss, matrix_records