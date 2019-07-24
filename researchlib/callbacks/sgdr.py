from .callback import Callback
from ..utils import *
import math


class SGDR(Callback):
    def __init__(self, step_size=1000, max_lr=1e-3):
        super().__init__()
        self.max_lr = max_lr
        self.base_lr = max_lr / 10
        self.length = 1
        self.step_size = 180.0 / step_size
        self.acc_iter = 0

    def on_iteration_begin(self, **kwargs):
        if kwargs['model'].training:
            eta = self.acc_iter / self.length
            if eta >= 180:
                self.length += 1
                self.acc_iter = 0
            val = eta % 180
            self.acc_iter += self.step_size
            cur_lr = self.base_lr + (1 + math.cos(math.radians(val))) / 2 * (
                self.max_lr - self.base_lr)
            set_lr(kwargs['optimizer'], cur_lr)
        return kwargs
