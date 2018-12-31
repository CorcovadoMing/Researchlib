from ..callbacks import *
import torch.nn.functional as F
from tqdm import tqdm

def train(model, train_loader, optimizer, epoch, callbacks=[]):
    return train_minibatch(model, train_loader, optimizer, epoch, callbacks)

def train_minibatch(model, train_loader, optimizer, epoch, callbacks):
    model.train()
    loss_history = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
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
        loss = F.nll_loss(output, target)
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