from ..callbacks import *
from tqdm.auto import tqdm

def train(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, keep_shape_, require_data_, callbacks=[]):
    return train_minibatch(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, keep_shape_, require_data_, callbacks)

def train_minibatch(model, train_loader, optimizer, loss_fn, epoch, is_cuda, require_long_, keep_shape_, require_data_, callbacks):
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
        
        loss_input = [output, target]
        if require_data_:
            loss_input.append(data)
            
        if keep_shape_:
            loss = loss_fn(*loss_input)
        else:
            loss = loss_fn(*list(map(lambda x: x.view(-1,), loss_input)))
            
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