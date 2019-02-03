def MultipleLoss(input, target, losses, loss_weights=None, size_average=True):
    if loss_weights:
        lw = loss_weights
    else:
        lw = [1]*len(losses)
    
    total_loss = None
    for i, loss_fn in enumerate(losses):
        if i == 0: 
            total_loss = loss_fn(input[i], target[i], size_average)
        else:
            total_loss += loss_fn(input[i], target[i], size_average)
    
    return total_loss