import torch
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
    bias_scale = parameter_manager.get_param('bias_scale')
    ema = parameter_manager.get_param('ema')
    ema_freq = parameter_manager.get_param('ema_freq')
    ema_momentum = parameter_manager.get_param('ema_momentum')
    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')
    rho = ema_momentum ** ema_freq    
    
    self.model.train()

    for m in metrics:
        m.reset()

    loss_record = 0
    norm_record = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Set LR
        cur_lr = Annealer.get_trace('lr')
        update_optim(self.optimizer, [cur_lr, cur_lr * bias_scale], key = 'lr')

        # Set weight decay
        if weight_decay > 0:
            cur_weight_decay = Annealer.get_trace('weight_decay')
            update_optim(self.optimizer, [cur_weight_decay, cur_weight_decay / bias_scale], key = 'weight_decay')
                
        if mmixup_alpha is not None:
            batch_size = inputs[0].size(0)
            if fixed_mmixup is None and random_mmixup is None:
                random_mmixup = [0, layer.ManifoldMixup.block_counter]
            lam = layer.ManifoldMixup.setup_batch(mmixup_alpha, batch_size, fixed_mmixup, random_mmixup)
            targets, targets_res = layer.ManifoldMixup.get_y(targets)
        else:
            targets_res = None
            lam = None

        for i in self.optimizer:
            i.zero_grad()
            
        if support_set is not None:
            support_x, support_y = support_set
            support_x = torch.from_numpy(support_x).to(inputs.device).to(inputs.dtype)
            support_x = support_x.view(-1, *inputs.shape[1:])
            outputs = self.model(inputs, support_x) # Batch, shot, way
            
            # Deal with labels
            support_y = torch.from_numpy(support_y).to(targets.device).to(targets.dtype)
            support_y = support_y.expand(targets.size(0), -1, -1).transpose(-1, -2) # Batch, shot, way
            targets = targets.unsqueeze(1).unsqueeze(2).expand(-1, shot, len(way))
            targets = targets.eq(support_y).to(inputs.dtype)
        else:
            outputs = self.model(inputs)
        
        loss = self.loss_fn[0](outputs, targets)
        loss.backward()
        
        for i in self.optimizer:
            i.step()    
        
        loss_record += loss.item()
        del loss
        
#         # May be a bottleneck for GPU utilization
#         for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
#             norm_record += p.grad.data.norm(2).item() ** 2

        if ema and (batch_idx + 1) % ema_freq == 0:
            for v, ema_v in zip(self.model.state_dict().values(), self.val_model.state_dict().values()):
                ema_v *= rho
                ema_v += (1-rho) * v


        for m in metrics:
            m.forward([outputs, targets])

        if batch_idx % 5 == 0:
            liveplot.update_desc(epoch, batch_idx + 1, loss_record / (batch_idx + 1), metrics, self.monitor)

        if batch_idx == (self.train_loader_length-1):
            liveplot.update_desc(epoch, batch_idx + 1, loss_record / (batch_idx + 1), metrics, self.monitor)
            break

        Annealer._iteration_step()

    loss_record = loss_record / (batch_idx + 1)
    norm_record = (norm_record ** 0.5) / (batch_idx + 1)
    return loss_record, norm_record