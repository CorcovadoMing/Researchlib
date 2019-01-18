from .train import *
from .test import *
from ..io import *
from ..callbacks import *
from ..utils import *
from ..metrics import *
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import *
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn

class FocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x, y):
        x = self.softmax(x)
        pt = x[torch.arange(x.size(0)), y]
        loss = -1 * ((1 - pt) ** self.gamma) * torch.log(pt)
        return loss.sum(-1).mean()
        
class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None):
        '''
            Multi-model supported
        '''
        self.is_cuda = torch.cuda.is_available()
        self.require_long_ = False
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.keep_x_shape_ = False
        self.keep_y_shape_ = False
        self.require_data_ = False
        self.multi_model = False
        
        self.default_callbacks = CyclicalLR(len(train_loader))
        
        self.trainer = train
        self.tester = test
        
        self.default_metrics = None
        
        if type(model) == type([]):
            self.multi_model = True
        
        # Check if cuda available (could be overwrite)
        if self.is_cuda:
            if self.multi_model:
                self.model = [i.cuda for i in model]
            else:
                self.model = model.cuda()
        else:
            self.model = model
            
        # Assign loss function
        if loss_fn == 'nll':
            self.loss_fn = F.nll_loss
            self.require_long_ = True
            self.default_metrics = Acc()
        elif loss_fn == 'crossentropy':
            self.loss_fn = nn.CrossEntropyLoss()
            self.require_long_ = True
            self.default_metrics = Acc()
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss()
            self.require_long_ = True
            self.default_metrics = Acc()
        elif loss_fn == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_fn == 'mae':
            self.loss_fn = F.l1_loss
        else:
            self.loss_fn = loss_fn
            self.keep_x_shape_ = True
            self.keep_y_shape_ = True
            try:
                self.require_data_ = loss_fn.require_data
            except:
                pass
            
        # Assign optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(model.parameters(), betas=(0.9, 0.99))
        elif optimizer == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(model.parameters())
        else:
            self.optimizer = optimizer
            
        cudnn.benchmark = True
        
    def set_cyclical_(self, lr):
        if self.default_callbacks:
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
        return [self.default_callbacks]
        
    def set_sgdr_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = SGDR(len(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            self.default_callbacks.length = 1
        return [self.default_callbacks]
    
    def set_onecycle_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = OneCycle(len(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
        return [self.default_callbacks]
    
    def fit_onecycle(self, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        callbacks = self.set_onecycle_(lr) + callbacks
        self.fit_(1, lr, augmentor, mixup_alpha, metrics, callbacks)
        
    def fit_cycle(self, cycles, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        total_epochs = int(cycles*(1+cycles)/2)
        callbacks = self.set_sgdr_(lr) + callbacks
        self.fit_(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks)
        
    def fit(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        callbacks = self.set_cyclical_(lr) + callbacks
        self.fit_(epochs, lr, augmentor, mixup_alpha, metrics, callbacks)

    def fit_(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics
        
        for epoch in tqdm(range(1, epochs + 1)):
            for callback_func in callbacks:
                callback_func.on_epoch_begin(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            self.trainer(model=self.model, 
                        train_loader=self.train_loader, 
                        optimizer=self.optimizer, 
                        loss_fn=self.loss_fn, 
                        epoch=epoch, 
                        augmentor=augmentor,
                        is_cuda=self.is_cuda, 
                        require_long=self.require_long_, 
                        keep_x_shape=self.keep_x_shape_,
                        keep_y_shape=self.keep_y_shape_,
                        require_data=self.require_data_,
                        mixup_alpha=mixup_alpha,
                        callbacks=callbacks,
                        metrics=metrics)
            
            for callback_func in callbacks:
                callback_func.on_epoch_end(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            if self.test_loader:
                self.tester(model=self.model, 
                            test_loader=self.test_loader, 
                            loss_fn=self.loss_fn, 
                            is_cuda=self.is_cuda, 
                            require_long=self.require_long_, 
                            require_data=self.require_data_, 
                            keep_x_shape=self.keep_x_shape_,
                            keep_y_shape=self.keep_y_shape_,
                            metrics=metrics)

            for callback_func in callbacks:
                callback_func.on_validation_end(model=self.model, 
                                                test_loader=self.test_loader,
                                                epoch=epoch)
            
    
    def validate(self, metrics=[]):
        '''
            Multi-model supported
        '''
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics
        
        self.tester(model=self.model, 
                    test_loader=self.test_loader, 
                    loss_fn=self.loss_fn, 
                    is_cuda=self.is_cuda, 
                    require_long=self.require_long_, 
                    require_data=self.require_data_, 
                    keep_x_shape=self.keep_x_shape_,
                    keep_y_shape=self.keep_y_shape_,
                    metrics=metrics)
    
    def save(self, path):
        '''
            Save model to path
            Multi-model supported
        '''
        save_model(self.model, path)
    
    def load(self, path):
        '''
            Load model from path
            Multi-model supported
        '''
        load_model(self.model, path)

    def find_lr(self, mixup_alpha=0, plot=False):
        '''
            Multi-model supported
        '''
        save_model(self.model, 'tmp.h5')
        try:
            loss = self.trainer(model=self.model, 
                                train_loader=self.train_loader, 
                                optimizer=self.optimizer, 
                                loss_fn=self.loss_fn, 
                                epoch=1,
                                augmentor=None,
                                is_cuda=self.is_cuda, 
                                require_long=self.require_long_, 
                                keep_x_shape=self.keep_x_shape_,
                                keep_y_shape=self.keep_y_shape_,
                                require_data=self.require_data_,
                                mixup_alpha=mixup_alpha,
                                callbacks=[LRRangeTest(len(self.train_loader), cutoff_ratio=10)],
                                metrics=[])
            
            step = (10 / 1e-9) ** (1 / len(self.train_loader))
            self.loss_history = []
            self.lr_history = []
            for i, j in enumerate(loss):    
                self.loss_history.append(j)
                self.lr_history.append(1e-9 * (step ** i))
            if plot:
                plot_utils(self.loss_history, self.lr_history)
        except:
            pass
        finally:
            load_model(self.model, 'tmp.h5')
