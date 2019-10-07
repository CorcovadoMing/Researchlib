from apex import amp
from ..utils import _register_method, ParameterManager, Annealer, set_lr, update_optim
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
def train_fn(self, loader, metrics, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    liveplot = parameter_manager.get_param('liveplot', required=True)
    mmixup_alpha = parameter_manager.get_param('mmixup_alpha')
    fixed_mmixup = parameter_manager.get_param('fixed_mmixup')
    random_mmixup = parameter_manager.get_param('random_mmixup')
    epoch = parameter_manager.get_param('epoch')
    warmup = parameter_manager.get_param('warmup')
    weight_decay = parameter_manager.get_param('weight_decay')
    
    self.model.train()

    for m in metrics:
        m.reset()

    loss_record = 0
    norm_record = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Set LR
        if epoch <= warmup:
            cur_lr = Annealer.get_trace('warmup_lr')
        else:
            cur_lr = Annealer.get_trace('regular_lr')
        set_lr(self.optimizer, cur_lr)

        # Set weight decay
        if weight_decay > 0:
            cur_weight_decay = Annealer.get_trace('weight_decay')
            update_optim(self.optimizer, cur_weight_decay, key = 'weight_decay')
                
        if mmixup_alpha is not None:
            batch_size = inputs[0].size(0)
            if fixed_mmixup is None and random_mmixup is None:
                random_mmixup = [0, layer.ManifoldMixup.block_counter]
            lam = layer.ManifoldMixup.setup_batch(mmixup_alpha, batch_size, fixed_mmixup, random_mmixup)
            targets, targets_res = layer.ManifoldMixup.get_y(targets)
        else:
            targets_res = None
            lam = None

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn[0](outputs, targets)
        loss.backward()
        self.optimizer.step()

#         # May be a bottleneck for GPU utilization
#         for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
#             norm_record += p.grad.data.norm(2).item() ** 2

        for m in metrics:
            m.forward([outputs, targets])

        loss_record += loss.item()
        
        # May be a bottleneck for GPU utilization
        liveplot.update_desc(epoch, batch_idx + 1, loss_record / (batch_idx + 1), metrics, self.monitor)

        if batch_idx == (self.train_loader_length-1):
            break

        Annealer._iteration_step()

    loss_record = loss_record / (batch_idx + 1)
    norm_record = (norm_record ** 0.5) / (batch_idx + 1)
    return loss_record, norm_record