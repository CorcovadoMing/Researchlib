import torch
import torch.nn.functional as F
from .history import *

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
        for data, target in kwargs['test_loader']:
            if kwargs['require_long']: target = target.long()
            if kwargs['is_cuda']: data, target = data.cuda(), target.cuda()
            
            kwargs['data'], kwargs['target'] = data, target
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_begin(**kwargs)
            
            if type(kwargs['data']) == type([]):
                output = kwargs['model'](*kwargs['data'])
            else:
                output = kwargs['model'](kwargs['data'])
            
            loss_input = [output, kwargs['target']]
            
            if not kwargs['keep_x_shape']: loss_input[0] = loss_input[0].contiguous().view(-1, loss_input[0].size(-1))
            
            if not kwargs['keep_y_shape']: 
                if len(loss_input[1].shape) > 1:
                    loss_input[1] = loss_input[1].contiguous().view(-1, loss_input[1].size(-1))
                else:
                    loss_input[1] = loss_input[1].contiguous().view(-1)
                    
            if kwargs['require_data']: loss_input.append(data)
            
            test_loss += kwargs['loss_fn'](*loss_input).item()
            
            # Capsule, TODO: refine the part
            if type(loss_input[0]) == type(()):
                loss_input[0] = loss_input[0][0]
                loss_input[0] = torch.sqrt((loss_input[0]**2).sum(dim=2, keepdim=True))
            
            for m in kwargs['metrics']: m.forward(loss_input)
            
            for callback_func in kwargs['callbacks']: kwargs = callback_func.on_iteration_end(**kwargs)

    # Output metrics
    for m in kwargs['metrics']: matrix_records.add(m.output(), prefix='val')    
    
    test_loss /= len(kwargs['test_loader'].dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    
    for callback_func in kwargs['callbacks']: kwargs = callback_func.on_validation_end(**kwargs)
    
    return test_loss, matrix_records