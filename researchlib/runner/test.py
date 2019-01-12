import torch
import torch.nn.functional as F

def test(model, test_loader, loss_fn, is_cuda, require_long, require_data, keep_x_shape, keep_y_shape, metrics):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Reset metrics
    for m in metrics: m.reset()
    
    with torch.no_grad():
        for data, target in test_loader:
            if require_long: target = target.long()
            if is_cuda: data, target = data.cuda(), target.cuda()
            
            output = model(data)
            
            loss_input = [output, target]
            
            if require_data: loss_input.append(data)
            if not keep_x_shape: loss_input[0] = loss_input[0].contiguous().view(-1, loss_input[0].size(-1))
            
            if not keep_y_shape: 
                loss_input[1] = loss_input[1].contiguous().view(-1)
            else:
                loss_input[1] = loss_input[1].contiguous().view(-1, loss_input[1].size(-1))
            
            test_loss += loss_fn(*loss_input).item()
            
            # Capsule
            if type(loss_input[0]) == type(()):
                loss_input[0] = loss_input[0][0]
                loss_input[0] = torch.sqrt((loss_input[0]**2).sum(dim=2, keepdim=True))
            
#             total += len(loss_input[1])
#             pred = loss_input[0].max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(loss_input[1].view_as(pred)).sum().item()
            for m in metrics: m.forward(output, target)
    
    # Output metrics
    for m in metrics: m.output()    
# test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#        test_loss, correct, total,
#        100. * correct / total))
#     print('\nTest set: Average loss: {:.4f}'.format(
#         test_loss))