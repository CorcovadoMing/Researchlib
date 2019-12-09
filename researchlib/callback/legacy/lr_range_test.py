from .callback import Callback
from ..utils import *


class LRRangeTest(Callback):
    def __init__(self, iterations, max_lr = 3, min_lr = 1e-9, cutoff_ratio = None):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cutoff_ratio = cutoff_ratio
        self.step = (self.max_lr / self.min_lr) ** (1 / float(iterations))

    def on_iteration_begin(self, **kwargs):
        cur_lr = self.min_lr * (self.step ** kwargs['batch_idx'])
        set_lr(kwargs['optimizer'], cur_lr)
        return kwargs

    def on_iteration_end(self, **kwargs):
        if kwargs['batch_idx'] == 0:
            self.cutoff_loss = kwargs['cur_loss'] * self.cutoff_ratio
        if kwargs['cur_loss'] > self.cutoff_loss:
            kwargs['check'].cutoff = True
        return kwargs
