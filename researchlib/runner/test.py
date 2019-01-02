import torch
import torch.nn.functional as F

def test(model, test_loader, loss_fn, is_cuda, require_long_):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if require_long_:
                target = target.long()
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_fn(output.view(-1, output.shape[-1]), target.view(-1,), reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))