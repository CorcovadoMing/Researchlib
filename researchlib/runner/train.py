from ..callbacks import *
from tqdm.auto import tqdm

def train(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, callbacks=[]):
    return train_minibatch(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, callbacks)

def train_minibatch(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, callbacks):
    model.train()
    loss_history = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if require_long_:
            target = target.long()
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()

        for callback_func in callbacks:
            callback_func.on_iteration_begin(model=model, 
                                            train_loader=train_loader, 
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            batch_idx=batch_idx,
                                            data=data,
                                            target=target)

        output = model(data)

        loss = loss_fn(output.view(-1, output.shape[-1]), target.view(-1,))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        for callback_func in callbacks:
            callback_func.on_iteration_end(model=model, 
                                            loss=loss, 
                                            train_loader=train_loader, 
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            batch_idx=batch_idx,
                                            data=data,
                                            target=target)
    return loss_history