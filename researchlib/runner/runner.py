from .train import train_fn
from .test import test_fn
from .history import *
from ..callbacks import *
from ..utils import *
from ..metrics import *
from ..loss import *
from ..layers import *
from .save_load import _save_model, _load_model
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import *
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from apex import amp
import torch.nn.init as init
from ..models import *


def _get_iteration(train_loader):
    iteration = None
    try:
        iteration = len(train_loader)
    except:
        iteration = (train_loader._size / train_loader.batch_size)
    assert iteration != None
    return iteration


class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None, reg_fn={}, reg_weights={}, monitor_mode='min', monitor_state='loss', fp16=False, multigpu=False):
        '''
            Multi-model supported
        '''
        self.is_cuda = torch.cuda.is_available()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history_ = History()
        self.multi_model = False
        self.cam_model = None
        
        self.default_callbacks = CyclicalLR(_get_iteration(self.train_loader))
        
        self.trainer = train_fn
        self.tester = test_fn
        
        self.default_metrics = None
        
        # Assign loss function
        # 
        # self.loss_fn
        # self.require_long_
        # self.keep_x_shape_
        # self.keep_y_shape_
        # self.default_metrics
        self.loss_fn = []
        self.require_long_ = []
        self.keep_x_shape_ = []
        self.keep_y_shape_ = []
        self.default_metrics = None
        # --------------------------------------------------------------------------------------------------------------------------------
        def _process_loss_fn(loss_fn):
            if type(loss_fn) == type({}):
                process_func = loss_ensemble
            else:
                process_func = loss_mapping
            return process_func(loss_fn)
        
        if type(loss_fn) == type([]):
            for lf in loss_fn:
                _loss_fn, require_long_, keep_x_shape_, keep_y_shape_, self.default_metrics = _process_loss_fn(lf)
                self.loss_fn.append(_loss_fn)
                self.require_long_ += require_long_
                self.keep_x_shape_ += keep_x_shape_
                self.keep_y_shape_ += keep_y_shape_
        else:
            _loss_fn, require_long_, keep_x_shape_, keep_y_shape_, self.default_metrics = _process_loss_fn(loss_fn)
            self.loss_fn.append(_loss_fn)
            self.require_long_ += require_long_
            self.keep_x_shape_ += keep_x_shape_
            self.keep_y_shape_ += keep_y_shape_
        # --------------------------------------------------------------------------------------------------------------------------------
        
        
        
        # Assign optimizer
        # 
        # self.optimizer
        # --------------------------------------------------------------------------------------------------------------------------------
        def _assign_optim(model, optimizer):
            if optimizer == 'adam':
                return Adam(model.parameters(), betas=(0.9, 0.99))
            elif optimizer == 'sgd':
                return SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-5)
            elif optimizer == 'rmsprop':
                return RMSprop(model.parameters(), weight_decay=5e-5)
            else:
                return optimizer
        
        if type(model) == GANModel:
            self.optimizer = [_assign_optim(model.discriminator, optimizer), _assign_optim(model.generator, optimizer)]
        else:
            self.optimizer = _assign_optim(model, optimizer)
        # --------------------------------------------------------------------------------------------------------------------------------
        
        
        
        
        # Assign monitoring
        #
        # self.monitor_mode
        # self.monitor_state
        # self.monitor
        self.monitor_mode = monitor_mode
        self.monitor_state = monitor_state
        self.monitor = None
        # --------------------------------------------------------------------------------------------------------------------------------
        if self.default_metrics:
            self.monitor_state = 'metrics'
            self.monitor_mode = 'max'
        # --------------------------------------------------------------------------------------------------------------------------------
        if monitor_mode == 'min':
            self.monitor = 1e9
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = 0
            self.monitor_mode = max
        # --------------------------------------------------------------------------------------------------------------------------------
        
        self.reg_fn = reg_fn
        for key in self.reg_fn:
            if type(reg_fn[key]) == type(''):
                fn, _, _, _, _ = loss_mapping(reg_fn[key])
                reg_fn[key] = fn
        
        self.reg_weights = reg_weights
        
        cudnn.benchmark = True
        
        # Model
        self.model = model
        if type(model) == GANModel:
            self.loss_fn[0].set_model(self.model)
        
        # FP16
        if self.is_cuda:
            self.model = self.model.cuda()
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", enabled=fp16)
        
        # Multi GPU
        self.multigpu = multigpu
        if self.multigpu:
            self.model = nn.DataParallel(self.model)
        
        
    def summary(self):
        input_shape = self.train_loader.dataset.train_data.shape
        if len(input_shape) > 4:
            input_shape = (input_shape[-1], input_shape[-3], input_shape[-2])
        else:
            input_shape = (1, input_shape[-2], input_shape[-1])
        try:
            summary(self.model, input_shape)
        except:
            pass
        
    def set_cyclical_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = CyclicalLR(_get_iteration(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            return [self.default_callbacks]
        else:
            return []
            
    def set_sgdr_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = SGDR(_get_iteration(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            self.default_callbacks.length = 1
            return [self.default_callbacks]
        else:
            return []
    
    def set_onecycle_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = OneCycle(_get_iteration(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            return [self.default_callbacks]
        else:
            return []
            
            
    def fit(self, epochs, lr=1e-3, cycle='default', augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        if cycle == 'sc' or cycle == 'superconverge':
            total_epochs = epochs
            callbacks = self.set_onecycle_(lr) + callbacks
        elif cycle == 'default':
            total_epochs = epochs 
            callbacks = self.set_cyclical_(lr) + callbacks
        elif cycle == 'cycle':
            total_epochs = int(epochs * (1 + epochs) / 2)
            callbacks = self.set_sgdr_(lr) + callbacks
        self.fit_(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks)
        

    def fit_(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
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
                if self.monitor_mode(cri, self.monitor) == cri:
                    self.monitor = cri
                    self.save('checkpoint.h5')
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
            
    
    
    def history(self, plot=True):
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(24, 8))
            legends = [[], []]
            for key in self.history_.records:
                if 'loss' in key:
                    legends[0].append(key)
                    ax[0].plot(self.history_.records[key])
                else:
                    legends[1].append(key)
                    ax[1].plot(self.history_.records[key])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epochs")
            ax[0].set_ylabel("Loss")
            ax[0].legend(legends[0])
            ax[1].set_title("Accuracy")
            ax[1].set_xlabel("Epochs")
            ax[1].set_ylabel("Accuracy")
            ax[1].legend(legends[1])
            plt.show()
        else:
            return self.history_
            
    
    def validate(self, metrics=[], callbacks=[]):
        '''
            Multi-model supported
        '''
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics
        
        loss_records, matrix_records = self.tester(model=self.model, 
                                                            test_loader=self.test_loader, 
                                                            loss_fn=self.loss_fn, 
                                                            is_cuda=self.is_cuda,
                                                            epoch=1,
                                                            require_long=self.require_long_, 
                                                            keep_x_shape=self.keep_x_shape_,
                                                            keep_y_shape=self.keep_y_shape_,
                                                            metrics=metrics,
                                                            callbacks=callbacks)
        if len(metrics) > 0: 
            print(loss_records, list(matrix_records.records.values())[-1][-1])
        else:
            print(loss_records)
    
    def init_model(self, init_distribution='xavier_normal', module_list=[]):    
        def _is_init_module(m, module_list):
            if len(module_list):
                if type(m) in module_list:
                    return True
                else:
                    return False
            else:
                return True
            
        def _init(m):
            if _is_init_module(m, module_list):
                for p in m.parameters():
                    if p.dim() > 1:
                        if init_distribution == 'xavier_normal':
                            init.xavier_normal_(p.data)
                        elif init_distribution == 'orthogonal':
                            init.orthogonal_(p.data)
                        print('Init ' + str(init_distribution) + ':', m)
                    else:
                        init.normal_(p.data)
        self.model.apply(_init)


    def save(self, path):
        _save_model(self.model, path)
    
    def load(self, path):
        self.model = _load_model(self.model, path, self.multigpu)

    def find_lr(self, mixup_alpha=0, plot=False, callbacks=[]):
        '''
            Multi-model supported
        '''
        _save_model(self.model, 'tmp.h5')
        try:
            loss, _ = self.trainer(model=self.model, 
                                train_loader=self.train_loader, 
                                optimizer=self.optimizer, 
                                loss_fn=self.loss_fn,
                                reg_fn=self.reg_fn,
                                reg_weights=self.reg_weights,
                                epoch=1,
                                augmentor=None,
                                is_cuda=self.is_cuda, 
                                require_long=self.require_long_, 
                                keep_x_shape=self.keep_x_shape_,
                                keep_y_shape=self.keep_y_shape_,
                                mixup_alpha=mixup_alpha,
                                callbacks=[LRRangeTest(_get_iteration(self.train_loader), cutoff_ratio=10)]+callbacks,
                                metrics=[])
            
            step = (10 / 1e-9) ** (1 / _get_iteration(self.train_loader))
            self.loss_history = []
            self.lr_history = []
            for i, j in enumerate(loss):    
                self.loss_history.append(j)
                self.lr_history.append(1e-9 * (step ** i))
            if plot:
                plot_utils(self.loss_history, self.lr_history)
        except Exception as e: print('Error:', e)
        finally: self.model = _load_model(self.model, 'tmp.h5', self.multigpu)
    
    def cam(self, vx, final_layer, out_filters, classes):
        if not self.cam_model:
            self.cam_model = nn.Sequential(
                    *list(self.model.children())[:final_layer+1], 
                    nn.Conv2d(out_filters, classes, 1), 
                    AdaptiveConcatPool2d(1),
                    nn.Conv2d(classes*2, classes, 1),
                    Flatten(),
                    nn.LogSoftmax(1)
                ).cuda()
            self.cam_feature_layer = -5
            module_trainable(self.cam_model[:self.cam_feature_layer], False)
            #self.fit_onecycle()
            r = Runner(self.cam_model, self.train_loader, self.test_loader, 'adam', 'nll', fp16=False)
            r.fit(10, 1e-3)
            
        self.cam_feature = SaveFeatures(self.cam_model[self.cam_feature_layer])
        py = self.cam_model(vx.cuda().float())
        py = F.softmax(py, dim=-1)
        py = py.detach().cpu().numpy()[0]
        feat = self.cam_feature.features[0].detach().cpu().numpy()
        feat = np.maximum(0, feat)
        f2 = np.dot(np.rollaxis(feat,0,3), py)
        f2 -= f2.min()
        f2 /= f2.max()
        dx = vx.cpu().numpy().transpose(0,2,3,1)[0]
        #import skimage
        plt.axis('off')
        #plt.imshow(dx)
        #ss = skimage.transform.resize(f2, dx.shape[:2])
        plt.imshow(f2, alpha=0.5, cmap='hot')
        module_trainable(self.model, True)
        self.cam_feature.remove()
