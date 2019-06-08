import os
from tqdm.auto import tqdm
from tqdm import tnrange
from ..callbacks import *
from ..utils import _register_method, _get_iteration, set_lr
from .history import History
from ..models import GANModel
from .save_load import _load_optimizer
from itertools import cycle

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
def fit_iteration(self, iteration, lr=1e-3, policy='cyclical', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[], _id='none', self_iterative=False):
    #TODO: save_model
    if policy == 'sc' or policy == 'superconverge':
        callbacks = self._set_onecycle(lr) + callbacks
    elif policy == 'cyclical':
        callbacks = self._set_cyclical(lr) + callbacks
    elif policy == 'cycle':
        callbacks = self._set_sgdr(lr) + callbacks
    elif policy == 'fixed':
        set_lr(self.optimizer, lr)
    self._fit(1, lr, augmentor, mixup_alpha, metrics, callbacks, _id, self_iterative, cycle=True, total=iteration)


@register_method
def preload_gpu(self):
    if self.is_cuda:
        if type(self.optimizer) == tuple or type(self.optimizer) == list:
            for optim in self.optimizer:
                state = optim.state_dict()['state']
                for key in state:
                    for attr in state[key]:
                        try:
                            state[key][attr] = state[key][attr].cuda()
                        except:
                            pass
        else:
            state = self.optimizer.state_dict()['state']
            for key in state:
                for attr in state[key]:
                    try:
                        state[key][attr] = state[key][attr].cuda()
                    except:
                        pass
        self.model.cuda()


@register_method
def unload_gpu(self):
    if self.is_cuda:
        if type(self.optimizer) == tuple or type(self.optimizer) == list:
            for optim in self.optimizer:
                state = optim.state_dict()['state']
                for key in state:
                    for attr in state[key]:
                        try:
                            state[key][attr] = state[key][attr].cuda()
                        except:
                            pass
        else:
            state = self.optimizer.state_dict()['state']
            for key in state:
                for attr in state[key]:
                    try:
                        state[key][attr] = state[key][attr].cuda()
                    except:
                        pass
        self.model.cpu()
        self._unload_data()
        torch.cuda.empty_cache()



@register_method
def fit_xy(self, data_pack, inputs, lr=1e-3, policy='cyclical', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[], _id='none', self_iterative=False, _auto_gpu=False, _train=True):
    # TODO: metrics, history, testing, save_model
    if policy == 'sc' or policy == 'superconverge':
        callbacks = self._set_onecycle(lr) + callbacks
    elif policy == 'cyclical':
        callbacks = self._set_cyclical(lr) + callbacks
    elif policy == 'cycle':
        callbacks = self._set_sgdr(lr) + callbacks
    elif policy == 'fixed':
        set_lr(self.optimizer, lr)
        
    if len(self.experiment_name) == 0:
        self.start_experiment('default')
    
    if _auto_gpu:
        self.preload_gpu()
    
    try:
        self._fit_xy(data_pack,
                    inputs,
                    augmentor, 
                    mixup_alpha, 
                    callbacks, 
                    metrics, 
                    loss_history=[], 
                    g_loss_history=[], 
                    d_loss_history=[], 
                    matrix_records=History(), 
                    bar=None,
                    train=_train)
    except:
        raise
    finally:
        if _auto_gpu:
            self.unload_gpu()        


@register_method
def fit(self, epochs, lr=1e-3, policy='cyclical', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[], _id='none', self_iterative=False):
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
    self._fit(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks, _id, self_iterative, cycle=False)


@register_method
def _process_type(self, data_pack, inputs):
    ''' INTERNAL_FUNCTION:
        Split the x and y in loader for training
        Move x and y to GPU if cuda is available
    '''
        
    if type(data_pack[0]) == dict: # DALI
        data, target = data_pack[0]['data'], data_pack[0]['label']
    else:
        data, target = data_pack[0:inputs], data_pack[inputs:]

    if type(data) != list and type(data) != tuple: data = [data]
    if type(target) != list and type(target) != tuple: target = [target]
    
    if type(data[0]) != torch.Tensor: data, target = [torch.from_numpy(i) for i in data], [torch.from_numpy(i) for i in target]
    
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
def _unload_data(self):
    del self.data, self.target, self.target_res
    torch.cuda.empty_cache()


@register_method
def _fit_xy(self, data_pack, inputs, augmentor, mixup_alpha, callbacks, metrics, loss_history, g_loss_history, d_loss_history, matrix_records, bar, train):
    self.data, self.target = self._process_type(data_pack, inputs)
    self.data, self.target, self.target_res = self._process_data(self.data, self.target, augmentor, mixup_alpha)
    self.model.train()
    self.trainer(train=train,
                model=self.model,
                data=self.data,
                target=self.target,
                target_res=self.target_res,
                optimizer=self.optimizer, 
                loss_fn=self.loss_fn,
                mixup_loss_fn=self.mixup_loss_fn,
                reg_fn=self.reg_fn,
                reg_weights=self.reg_weights,
                epoch=self.epoch, 
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
    self.model.eval()


@register_method
def _cycle(self, data, _cycle_flag=False):
    if _cycle_flag:
        return cycle(enumerate(data))
    else:
        return enumerate(data)
        

@register_method
def _fit(self, epochs, lr, augmentor, mixup_alpha, metrics, callbacks, _id, self_iterative, cycle, total=0):
    if total == 0:
        total = len(self.train_loader)
        
    if len(self.experiment_name) == 0:
        self.start_experiment('default')
    
    self.preload_gpu()
    
    try:
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics

        for epoch in tqdm(range(1, epochs + 1)):
            for callback_func in callbacks:
                callback_func.on_epoch_begin(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=self.epoch)

            loss_history = []
            g_loss_history = []
            d_loss_history = []
            matrix_records = History()

            for m in metrics: m.reset()
            bar = tqdm(range(total), leave=False, initial=1)

            iteration_break = total
            for batch_idx, data_pack in self._cycle(self.train_loader, cycle):
                self._fit_xy(data_pack,
                            self.inputs,
                            augmentor, 
                            mixup_alpha, 
                            callbacks, 
                            metrics, 
                            loss_history, 
                            g_loss_history, 
                            d_loss_history, 
                            matrix_records, 
                            bar,
                            train=True)
                            
                bar.update(1)
                iteration_break -= 1
                if iteration_break == 0:
                    bar.close()
                    break
                
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
                                                            epoch=self.epoch,
                                                            require_long=self.require_long_,  
                                                            keep_x_shape=self.keep_x_shape_,
                                                            keep_y_shape=self.keep_y_shape_,
                                                            metrics=metrics,
                                                            callbacks=callbacks,
                                                            inputs=self.inputs)

                self.history_.add(loss_records, prefix='val')
                self.history_ += matrix_records

            epoch_str = str(self.epoch)

            monitor_target = 'val_' + self.monitor_state if self.test_loader else 'train_' + self.monitor_state
            if monitor_target in self.history_.records:
                critic = self.history_.records[monitor_target][-1]
            else:
                critic = None

            # Checkpoint
            checkpoint_model_name = os.path.join(self.checkpoint_path, 'checkpoint_' + _id + '_epoch_' + str(self.epoch))
            self.save(checkpoint_model_name)
            if critic is not None and self.monitor_mode(critic, self.monitor) == critic:
                self.monitor = critic
                best_checkpoint_model_name = os.path.join(self.checkpoint_path, 'best_' + _id)
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

            self.epoch += 1
            
            # Self-interative
            if self_iterative:
                self.model.eval()
                for i in tnrange(len(self.train_loader.dataset.tensors[0])):
                    self.train_loader.dataset.tensors[1][i] = \
                    self.model(self.train_loader.dataset.tensors[0][i].unsqueeze(0).cuda()).detach().cpu()[0]
                    torch.cuda.empty_cache()
                
                
            
    except:
        raise
    
    finally:
        self.unload_gpu()
        