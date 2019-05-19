import os
from tqdm.auto import tqdm
from ..callbacks import *
from ..utils import _register_method, _get_iteration, set_lr

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
def fit(self, epochs, lr=1e-3, policy='cyclical', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
    if policy == 'sc' or policy == 'superconverge':
        total_epochs = epochs
        callbacks = self._set_onecycle(lr) + callbacks
    elif policy == 'cyclical':
        total_epochs = epochs 
        callbacks = self._set_cyclical(lr) + callbacks
    elif policy == 'cycle':
        total_epochs = int(epochs * (1 + epochs) / 2)
        callbacks = self._set_sgdr(lr) + callbacks
    elif policy == 'fixed':
        set_lr(self.optimizer, lr)
        total_epochs = epochs
    self._fit(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks)

@register_method
def _fit(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
    if len(self.experiment_name) == 0:
        self.start_experiment('default')
        
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

        
        monitor_target = 'val_' + self.monitor_state if self.test_loader else 'train_' + self.monitor_state
        if monitor_target in self.history_.records:
            critic = self.history_.records[monitor_target][-1]
        else:
            critic = None

        # Checkpoint
        checkpoint_model_name = os.path.join(self.checkpoint_path, 'checkpoint_model_epoch_' + str(epoch) + '.h5')
        self.save(checkpoint_model_name)
        if critic is not None and self.monitor_mode(critic, self.monitor) == critic:
            self.monitor = critic
            best_checkpoint_model_name = os.path.join(self.checkpoint_path, 'best.h5')
            self.save(best_checkpoint_model_name)
            epoch_str += '*'
            self.history_.add({'saved': '*'})
        else:
            self.history_.add({'saved': ''})

        state = []
        fs = '{:^14}'
        if epoch == 1:
            print(('{:^10}' + (fs * (len(self.history_.records.keys()) - 1))).format('Epochs', *list(self.history_.records.keys())[:-1]))
            if self.test_loader:
                print('================================================================')
            else:
                print('==============================')
        state.append('{:^10}'.format(epoch_str))
        for i in self.history_.records:
            if i != 'saved': state.append('{:^14.4f}'.format(self.history_.records[i][-1]))
        print(''.join(state))
