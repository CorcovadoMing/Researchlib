import os
from tqdm.auto import tqdm
from ..callbacks import *
from ..utils import _register_method, _get_iteration, set_lr
from .history import History
from ..models import GANModel

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
def _process_type(self, data_pack):
    ''' INTERNAL_FUNCTION:
        Split the x and y in loader for training
        Move x and y to GPU if cuda is available
    '''
        
    if type(data_pack[0]) == dict: # DALI
        data, target = data_pack[0]['data'], data_pack[0]['label']
    else:
        data, target = data_pack[0:self.inputs], data_pack[self.inputs:]

    if type(data) != type([]) and type(data) != type(()): data = [data]
    if type(target) != type([]) and type(target) != type(()): target = [target]
    
    # GPU
    if self.is_cuda: data, target = [i.cuda() for i in data], [i.cuda() for i in target]
    
    return data, target
    
@register_method
def _process_data(self, data, target, augmentor, mixup_alpha):
    ''' INTERNAL_FUNCTION:
        Augmentation and Mixup
    '''
    
    def mixup_loss_fn(loss_fn, x, y, y_res, lam):
        return lam * loss_fn(x, y) + (1 - lam) * loss_fn(x, y_res)

    
    # On the fly augmentation
    if augmentor: data, target = augmentor.on(data, target)

    # Target type refine
    target = [i.long() if j else i for i, j in zip(target, self.require_long_)]

    # Mixup
    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        index = torch.randperm(data[0].size(0))
        data[0] = lam * data[0] + (1-lam) * data[0][index]
        target_res = [i[index] for i in target]
        if self.is_cuda: target_res = [i.cuda() for i in target_res]
        self.lam = lam
        self.mixup_loss_fn = mixup_loss_fn
    else:
        self.lam = None
        self.mixup_loss_fn = None
        target_res = None
    
    return data, target, target_res

@register_method
def _unload_data(self, data, target, target_res):
    del data, target, target_res
    torch.cuda.empty_cache()

@register_method
def _fit(self, epochs, lr, augmentor, mixup_alpha, metrics, callbacks):
    self.set_optimizer('adam')
    self.model.cuda()


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

        loss_history = []
        g_loss_history = []
        d_loss_history = []
        matrix_records = History()
        
        for m in metrics: m.reset()
        bar = tqdm(self.train_loader, leave=False)
        
        for batch_idx, data_pack in enumerate(bar):
            data, target = self._process_type(data_pack)
            data, target, target_res = self._process_data(data, target, augmentor, mixup_alpha)
            
            self.trainer(model=self.model,
                        data=data,
                        target=target,
                        target_res=target_res,
                        optimizer=self.optimizer, 
                        loss_fn=self.loss_fn,
                        mixup_loss_fn=self.mixup_loss_fn,
                        reg_fn=self.reg_fn,
                        reg_weights=self.reg_weights,
                        epoch=epoch, 
                        keep_x_shape=self.keep_x_shape_,
                        keep_y_shape=self.keep_y_shape_,
                        mixup_alpha=mixup_alpha,
                        callbacks=callbacks,
                        metrics=metrics,
                        loss_history=loss_history,
                        g_loss_history=g_loss_history,
                        d_loss_history=d_loss_history,
                        matrix_records=matrix_records,
                        bar=bar)

        self._unload_data(data, target, target_res)
        
        # Output metrics
        for m in metrics: matrix_records.add(m.output(), prefix='train')
        if type(self.model) == GANModel:
            loss_records = {'d_loss': sum(d_loss_history)/len(d_loss_history), 'g_loss': sum(g_loss_history)/len(g_loss_history)}
        else:
            loss_records = {'loss': sum(loss_history)/len(loss_history)}

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
                                                        callbacks=callbacks,
                                                        inputs=self.inputs)

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
        
        self.model.cpu()
        del self.optimizer
        torch.cuda.empty_cache()
