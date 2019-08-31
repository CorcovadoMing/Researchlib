import torch
from .history import *
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def validate_fn(self, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    metrics = parameter_manager.get_param('metrics', [])
    callbacks = parameter_manager.get_param('callbacks', [])
    test_loader = parameter_manager.get_param('test_loader', required=True)
    loss_fn = parameter_manager.get_param('loss_fn', required=True)
    
    self.model.eval()
    matrix_records = History()

    if self.swa and self.epoch >= self.swa_start:
        _switch_swa_mode(self.optimzier)
        if type(self.optimizer) == list:
            i.bn_update(self.train_loader, self.model, device='cuda')
        else:
            self.optimizer.bn_update(self.train_loader, self.model, device='cuda')

    # Reset metrics
    for m in metrics: 
        m.reset()

    test_loss = 0
    total = len(self.test_loader)
    with torch.no_grad():
        for batch_idx, (x, y, _, _) in enumerate(test_loader):
            if self.is_cuda: x, y = [i.cuda() for i in x], [i.cuda() for i in y]

            for callback_func in callbacks:
                kwargs = callback_func.on_iteration_begin(**kwargs)

            output = self.model(*x)
            auxout = get_aux_out(self.model)
            auxout.append(output)
            auxout = [i.view(i.size(0), -1) for i in auxout]

            y = [i.view(i.size(0), -1) for i in y]
            if len(y) > len(auxout):
                y = [y]

            for i in range(len(auxout)):
                test_loss += loss_fn[i](auxout[i], y[i]).item()

            for m in metrics:
                m.forward([auxout[-1], y[-1]])

            for callback_func in callbacks:
                kwargs = callback_func.on_iteration_end(**kwargs)

            if batch_idx + 1 == total:
                break

    test_loss /= total

    for callback_func in callbacks:
        kwargs = callback_func.on_validation_end(**kwargs)

    # Output metrics
    for m in metrics:
        matrix_records.add(m.output(), prefix='val')

    if self.swa and self.epoch >= self.swa_start:
        _switch_swa_mode(self.optimzier)

    return {'loss': test_loss}, matrix_records
