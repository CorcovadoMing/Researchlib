import torch
from .history import *
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager
from functools import reduce

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def validate_fn(self, loader, metrics):
    self.model.eval()

    for m in metrics:
        m.reset()

    loss_record = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            outputs = self.model(inputs)
            loss = self.loss_fn[0](outputs, targets)

            for m in metrics:
                m.forward([outputs, targets])

            loss_record += loss.item()

            if batch_idx == (self.test_loader_length-1):
                break

    loss_record = loss_record / (batch_idx + 1)
    return loss_record