import torch
import torch.nn.functional as F
from .history import *
from ..utils import _register_method, get_aux_out

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def validate_fn(self, **kwargs):
    self.model.eval()
    test_loss = 0
    matrix_records = History()

    last_acc_val = self.history_.records['val_acc'][
        -1] if 'val_acc' in self.history_.records else 0.
    if self.swa and (self.epoch >= self.swa_start
                     or last_acc_val >= self.swa_val_acc):
        if type(self.optimizer) == list:
            for i in self.optimizer:
                i.swap_swa_sgd()
                i.bn_update(self.train_loader, self.model, device='cuda')
        else:
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader,
                                     self.model,
                                     device='cuda')

    # Reset metrics
    for m in kwargs['metrics']:
        m.reset()

    total = len(self.test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(kwargs['test_pipe_generator']):
            kwargs['batch_idx'] = batch_idx

            if kwargs['is_cuda']:
                x, y = [i.cuda() for i in x], [i.cuda() for i in y]
            kwargs['data'], kwargs['target'] = x, y

            for callback_func in kwargs['callbacks']:
                kwargs = callback_func.on_iteration_begin(**kwargs)

            output = self.model(*x)

            auxout = get_aux_out(self.model)
            auxout.append(output)
            kwargs['auxout'] = auxout

            auxout = [i.view(i.size(0), -1) for i in auxout]

            kwargs['target'] = [
                i.view(i.size(0), -1) for i in kwargs['target']
            ]
            if len(kwargs['target']) > len(auxout):
                kwargs['target'] = [kwargs['target']]

            for i in range(len(auxout)):
                test_loss += kwargs['loss_fn'][i](auxout[i],
                                                  kwargs['target'][i]).item()

            for m in kwargs['metrics']:
                m.forward([auxout[-1]] + [kwargs['target'][-1]])

            for callback_func in kwargs['callbacks']:
                kwargs = callback_func.on_iteration_end(**kwargs)

            if batch_idx + 1 == total:
                break

    test_loss /= (kwargs['batch_idx'] + 1)

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_validation_end(**kwargs)

    # Output metrics
    for m in kwargs['metrics']:
        matrix_records.add(m.output(), prefix='val')

    if self.swa and (self.epoch >= self.swa_start
                     or last_acc_val >= self.swa_val_acc):
        if type(self.optimizer) == list:
            for i in self.optimizer:
                i.swap_swa_sgd()
        else:
            self.optimizer.swap_swa_sgd()

    return {'loss': test_loss}, matrix_records
