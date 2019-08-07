import torch
import torch.nn.functional as F
from .history import *
from ..utils import *
import torchtext
from ..utils import _register_method

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def validate_fn(self, **kwargs):
    if self.swa and self.epoch >= self.swa_start:
        if type(self.optimizer) == list:
            for i in self.optimizer:
                i.swap_swa_sgd()
                i.bn_update(self.train_loader, self.model, device='cuda')
        else:
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.model, device='cuda')
        
    kwargs['model'].eval()
    test_loss = 0
    matrix_records = History()

    # Reset metrics
    for m in kwargs['metrics']:
        try:
            m.reset()
        except:
            pass

    with torch.no_grad():
        for batch_idx, data_pack in enumerate(kwargs['test_loader']):
            if type(kwargs['test_loader']
                    ) == torchtext.data.iterator.BucketIterator:
                data, target = data_pack.text, data_pack.label
            elif type(data_pack[0]) == type({}):
                data, target = data_pack[0]['data'], data_pack[0]['label']
            else:
                data, target = data_pack[0], data_pack[1:]

            kwargs['batch_idx'] = batch_idx

            if type(data) != type([]) and type(data) != type(()): data = [data]
            if type(target) != type([]) and type(target) != type(()):
                target = [target]
                
            # Preprocessing (experimental)
            for preprocessing_fn in self.preprocessing_list:
                data, target = preprocessing_fn._forward(data, target)

            target = [
                i.long() if j else i
                for i, j in zip(target, kwargs['require_long'])
            ]

            if kwargs['is_cuda']:
                data, target = [i.cuda()
                                for i in data], [i.cuda() for i in target]

            kwargs['data'], kwargs['target'] = data, target

            for callback_func in kwargs['callbacks']:
                kwargs = callback_func.on_iteration_begin(**kwargs)

            output = kwargs['model'](*kwargs['data'])

            auxout = get_aux_out(kwargs['model'])
            auxout.append(output)
            kwargs['auxout'] = auxout

            auxout = [
                i if j else i.view(i.size(0), -1)
                for i, j in zip(auxout, kwargs['keep_x_shape'])
            ]
            kwargs['target'] = [
                i if j else i.view(i.size(0), -1)
                for i, j in zip(kwargs['target'], kwargs['keep_y_shape'])
            ]

            if len(kwargs['target']) > len(auxout):
                kwargs['target'] = [kwargs['target']]

            for i in range(len(auxout)):
                test_loss += kwargs['loss_fn'][i](auxout[i],
                                                  kwargs['target'][i]).item()

            for m in kwargs['metrics']:
                try:
                    m.forward([auxout[-1]] + [kwargs['target'][-1]])
                except:
                    pass

            for callback_func in kwargs['callbacks']:
                kwargs = callback_func.on_iteration_end(**kwargs)

    test_loss /= (kwargs['batch_idx'] + 1)

    for callback_func in kwargs['callbacks']:
        kwargs = callback_func.on_validation_end(**kwargs)

    # Output metrics
    for m in kwargs['metrics']:
        try:
            matrix_records.add(m.output(), prefix='val')
        except:
            pass

    if self.swa and self.epoch >= self.swa_start:
        if type(self.optimizer) == list:
            for i in self.optimizer:
                i.swap_swa_sgd()
        else:
            self.optimizer.swap_swa_sgd()
        
    return {'loss': test_loss}, matrix_records
