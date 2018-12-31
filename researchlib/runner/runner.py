from .train import *
from .test import *
from ..io import *
from ..callbacks import *
from ..utils import *

class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.model = model
    
    def set_optim_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def fit(self, epochs, lr=0.001, callbacks=[]):
        self.set_optim_lr(lr)
        for epoch in range(1, epochs + 1):
            train(self.model, self.train_loader, self.optimizer, epoch, callbacks)
            test(self.model, self.test_loader)
    
    def val(self):
        test(self.model, self.test_loader)
    
    def save(self, path):
        save_model(self.model, path)
    
    def load(self, path):
        load_model(self.model, path)

    def find_lr(self, plot=False):
        save_model(self.model, 'tmp.h5')
        loss = train(self.model, self.train_loader, self.optimizer, epoch=1, callbacks=[LRRangeTest(len(self.train_loader))])
        load_model(self.model, 'tmp.h5')
        
        step = (10 / 1e-5) ** (1 / len(self.train_loader))
        
        self.loss_history = []
        self.lr_history = []
        best = 1e9
        for i, j in enumerate(loss):
            if j > best*5:
                break
            if j < best:
                best = j
            self.loss_history.append(j)
            self.lr_history.append(1e-5 * (step ** i))
        
        if plot:
            plot_utils(self.loss_history, self.lr_history)
