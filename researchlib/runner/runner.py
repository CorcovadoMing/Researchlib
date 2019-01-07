from .train import *
from .test import *
from ..io import *
from ..callbacks import *
from ..utils import *
import torch
import torch.nn.functional as F
from torch.optim import *
from tqdm.auto import tqdm

class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None):
        self.is_cuda = torch.cuda.is_available()
        self.require_long_ = False
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.keep_shape_ = False
        self.require_data_ = False
        
        # Check if cuda available (could be overwrite)
        if self.is_cuda:
            self.model = model.cuda()
        else:
            self.model = model
            
        # Assign loss function
        if loss_fn == 'nll':
            self.loss_fn = F.nll_loss
            self.require_long_ = True
        elif loss_fn == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_fn == 'mae':
            self.loss_fn = F.l1_loss
        else:
            self.loss_fn = loss_fn
            self.keep_shape_ = True
            try:
                self.require_data_ = loss_fn.require_data
            except:
                pass
            
        # Assign optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(model.parameters())
        elif optimizer == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=1e-3)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(model.parameters())
        else:
            self.optimizer = optimizer


    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def fit(self, epochs, lr=0.001, callbacks=[]):
        self.set_lr(lr)
        for epoch in tqdm(range(1, epochs + 1)):
            for callback_func in callbacks:
                callback_func.on_epoch_begin(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            train(self.model, 
                    self.train_loader, 
                    self.optimizer, 
                    self.loss_fn, 
                    epoch, 
                    self.is_cuda, 
                    self.require_long_, 
                    self.keep_shape_,
                    self.require_data_,
                    callbacks)
            
            for callback_func in callbacks:
                callback_func.on_epoch_end(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            if self.test_loader:
                test(self.model, self.test_loader, self.loss_fn, self.is_cuda, self.require_long_, self.require_data_, self.keep_shape_)
    
    def val(self):
        test(self.model, self.test_loader, self.loss_fn, self.is_cuda, self.require_long_, self.require_data_, self.keep_shape_)
    
    def save(self, path):
        save_model(self.model, path)
    
    def load(self, path):
        load_model(self.model, path)

    def find_lr(self, plot=False):
        save_model(self.model, 'tmp.h5')
        loss = train(self.model, 
                    self.train_loader, 
                    self.optimizer, 
                    self.loss_fn, 
                    1, 
                    self.is_cuda, 
                    self.require_long_, 
                    self.keep_shape_,
                    self.require_data_,
                    callbacks=[LRRangeTest(len(self.train_loader))])
        load_model(self.model, 'tmp.h5')
        
        step = (10 / 1e-5) ** (1 / len(self.train_loader))
        
        self.loss_history = []
        self.lr_history = []
        start_loss = loss[0]*1.1
        for i, j in enumerate(loss):    
            if j > start_loss:
                break
            self.loss_history.append(j)
            self.lr_history.append(1e-5 * (step ** i))
        
        if plot:
            plot_utils(self.loss_history, self.lr_history)
