from ..callbacks import *
import torch
from torch.nn.utils import *
from apex import amp
from torch import nn
from ..models import GANModel, VAEModel
from ..utils import *
from ..utils import _register_method
from ..layers import layer

__methods__ = []
register_method = _register_method(__methods__)


def _backup_grad(model):
    for param in model.parameters():
        try:
            param.backup_grad = param.grad.data.clone()
        except:
            if param.requires_grad:
                param.backup_grad = param.data.clone()
                param.backup_grad.zero_()


def _restore_grad(model):
    for param in model.parameters():
        try:
            param.grad = param.backup_grad.data.clone()
        except:
            pass


@register_method
def train_fn(self, epoch, loader, metrics, liveplot, mmixup_alpha, fixed_mmixup, random_mmixup):
    self.model.train()

    for m in metrics:
        m.reset()

    loss_record = 0
    norm_record = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if mmixup_alpha is not None:
            batch_size = inputs[0].size(0)
            if fixed_mmixup is None and random_mmixup is None:
                random_mmixup = [0, layer.ManifoldMixup.block_counter]
            lam = layer.ManifoldMixup.setup_batch(mmixup_alpha, batch_size, fixed_mmixup, random_mmixup)
            targets, targets_res = layer.ManifoldMixup.get_y(targets)
            targets, targets_res = targets.cuda(), targets_res.cuda()
        else:
            targets_res = None
            lam = None


        inputs, targets = inputs.cuda(), targets.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn[0](outputs, targets)
        loss.backward()
        self.optimizer.step()

        # May be a bottleneck for GPU utilization
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            norm_record += p.grad.data.norm(2).item() ** 2

        for m in metrics:
            m.forward([outputs, targets])

        loss_record += loss.item()
        # May be a bottleneck for GPU utilization
        liveplot.update_progressbar(batch_idx + 1)
        liveplot.update_desc(epoch, loss_record / (batch_idx + 1), metrics, self.monitor)

        if batch_idx == (self.train_loader_length-1):
            break

        Annealer._iteration_step()

    loss_record = loss_record / (batch_idx + 1)
    norm_record = (norm_record ** 0.5) / (batch_idx + 1)
    return loss_record, norm_record