def text_log(obj):
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        obj['epoch'], 
        obj['batch_idx'] * len(obj['data']), 
        len(obj['train_loader'].dataset),
        100. * obj['batch_idx'] / len(obj['train_loader']), 
        obj['loss'].item()
        ))