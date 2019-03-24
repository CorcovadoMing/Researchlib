from .callback import Callback
from ..utils import *
import math

class OneCycle(Callback):
    def __init__(self, step_size=1000, max_lr=1e-3):
        super().__init__()
        self.max_lr = max_lr
        self.base_lr = max_lr / 10
        self.step_size = int(0.45*step_size)
        self.max_step_size = int(0.9*step_size)
        self.acc_iter = 0
        
    def on_iteration_begin(self, **kwargs):
        if kwargs['model'].training:
            if self.acc_iter < self.max_step_size:
                cycle = math.floor(1 + self.acc_iter / (2 * (self.step_size + 1)))
                x = abs(self.acc_iter / (self.step_size + 1) - 2 * cycle + 1)
                cur_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
                self.acc_iter += 1
            else:
                cur_lr = self.base_lr
            set_lr(kwargs['optimizer'], cur_lr)
        return kwargs
    
    def on_epoch_end(self, **kwargs):
        self.acc_iter = 0