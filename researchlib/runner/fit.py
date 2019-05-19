import os
from tqdm.auto import tqdm
from ..callbacks import *
from ..utils import _register_method, _get_iteration

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def _set_cyclical(self, lr):
    if self.default_callbacks:
        self.default_callbacks = CyclicalLR(_get_iteration(self.train_loader))
        self.default_callbacks.max_lr = lr
        self.default_callbacks.acc_iter = 0
        return [self.default_callbacks]
    else:
        return []

@register_method
def _set_sgdr(self, lr):
    if self.default_callbacks:
        self.default_callbacks = SGDR(_get_iteration(self.train_loader))
        self.default_callbacks.max_lr = lr
        self.default_callbacks.acc_iter = 0
        self.default_callbacks.length = 1
        return [self.default_callbacks]
    else:
        return []

@register_method
def _set_onecycle(self, lr):
    if self.default_callbacks:
        self.default_callbacks = OneCycle(_get_iteration(self.train_loader))
        self.default_callbacks.max_lr = lr
        self.default_callbacks.acc_iter = 0
        return [self.default_callbacks]
    else:
        return []

@register_method
def fit(self, epochs, lr=1e-3, cycle='default', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
    if cycle == 'sc' or cycle == 'superconverge':
        total_epochs = epochs
        callbacks = self._set_onecycle(lr) + callbacks
    elif cycle == 'default':
        total_epochs = epochs 
        callbacks = self._set_cyclical(lr) + callbacks
    elif cycle == 'cycle':
        total_epochs = int(epochs * (1 + epochs) / 2)
        callbacks = self._set_sgdr(lr) + callbacks
    self._fit(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks)

@register_method
def _fit(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
    if self.default_metrics:
        metrics = [self.default_metrics] + metrics

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_str = str(epoch)

        for callback_func in callbacks:
            callback_func.on_epoch_begin(model=self.model, 
                                        train_loader=self.train_loader, 
                                        optimizer=self.optimizer,
                                        epoch=epoch)

        loss_records, matrix_records = self.trainer(model=self.model, 
                                                    train_loader=self.train_loader, 
                                                    optimizer=self.optimizer, 
                                                    loss_fn=self.loss_fn,
                                                    reg_fn=self.reg_fn,
                                                    reg_weights=self.reg_weights,
                                                    epoch=epoch, 
                                                    augmentor=augmentor,
                                                    is_cuda=self.is_cuda, 
                                                    require_long=self.require_long_, 
                                                    keep_x_shape=self.keep_x_shape_,
                                                    keep_y_shape=self.keep_y_shape_,
                                                    mixup_alpha=mixup_alpha,
                                                    callbacks=callbacks,
                                                    metrics=metrics)

        self.history_.add(loss_records, prefix='train')
        self.history_ += matrix_records


        for callback_func in callbacks:
            callback_func.on_epoch_end(model=self.model, 
                                        train_loader=self.train_loader, 
                                        optimizer=self.optimizer,
                                        epoch=epoch)

        if self.test_loader:
            loss_records, matrix_records = self.tester(model=self.model, 
                                                        test_loader=self.test_loader, 
                                                        loss_fn=self.loss_fn, 
                                                        is_cuda=self.is_cuda,
                                                        epoch=epoch,
                                                        require_long=self.require_long_,  
                                                        keep_x_shape=self.keep_x_shape_,
                                                        keep_y_shape=self.keep_y_shape_,
                                                        metrics=metrics,
                                                        callbacks=callbacks)

            self.history_.add(loss_records, prefix='val')
            self.history_ += matrix_records

            cri = None
            if self.monitor_state == 'metrics':
                cri = list(matrix_records.records.values())[-1][-1]
            else:
                cri = [loss_records[key] for key in loss_records][-1]

            # Checkpoint
            checkpoint_model_name = os.path.join(self.checkpoint_path, 'checkpoint_model_epoch_' + str(epoch) + '.h5')
            self.save(checkpoint_model_name)
            if self.monitor_mode(cri, self.monitor) == cri:
                self.monitor = cri
                best_checkpoint_model_name = os.path.join(self.checkpoint_path, 'best.h5')
                self.save(best_checkpoint_model_name)
                epoch_str += '*'

        state = []
        fs = '{:^14}'
        if epoch == 1:
            if self.test_loader:
                print(('{:^10}'+(fs*len(self.history_.records.keys()))).format('Epochs', *self.history_.records.keys()))
                print('================================================================')
            else:
                print(('{:^10}'+(fs*len(self.history_.records.keys()))).format('Epochs', *self.history_.records.keys()))
                print('==============================') # Untested
        state.append('{:^10}'.format(epoch_str))
        for i in self.history_.records:
            state.append('{:^14.4f}'.format(self.history_.records[i][-1]))
        print(''.join(state))
