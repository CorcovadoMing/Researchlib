import torch
import torch.nn.functional as F

def test(**kwargs):
    '''
        kwargs: model, test_loader, loss_fn, is_cuda, require_long, require_data, keep_x_shape, keep_y_shape, metrics
    '''
    
    kwargs['model'].eval()
    test_loss = 0
    
    # Reset metrics
    for m in kwargs['metrics']: m.reset()
    
    with torch.no_grad():
        for data, target in kwargs['test_loader']:
            if kwargs['require_long']: target = target.long()
            if kwargs['is_cuda']: data, target = data.cuda(), target.cuda()
            
            output = kwargs['model'](data)
            
            loss_input = [output, target]
            
            if kwargs['require_data']: loss_input.append(data)
            if not kwargs['keep_x_shape']: loss_input[0] = loss_input[0].contiguous().view(-1, loss_input[0].size(-1))
            
            if not kwargs['keep_y_shape']: 
                loss_input[1] = loss_input[1].contiguous().view(-1)
            else:
                loss_input[1] = loss_input[1].contiguous().view(-1, loss_input[1].size(-1))
            
            test_loss += kwargs['loss_fn'](*loss_input).item()
            
            # Capsule, TODO: refine the part
            if type(loss_input[0]) == type(()):
                loss_input[0] = loss_input[0][0]
                loss_input[0] = torch.sqrt((loss_input[0]**2).sum(dim=2, keepdim=True))
            
            for m in kwargs['metrics']: m.forward(loss_input)
    
    # Output metrics
    for m in kwargs['metrics']: m.output()    
    
    test_loss /= len(kwargs['test_loader'].dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    