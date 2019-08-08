from ..utils import _register_method, _get_iteration, set_lr, plot_montage
from ..callbacks import *

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def _set_policy(self, policy, lr, callbacks=[], epochs=1):
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
    return total_epochs, callbacks

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