import torch
from torch._six import inf
from ..utils import _register_method, ParameterManager, update_optim
from ..ops import op
from ..models import Builder


__methods__ = []
register_method = _register_method(__methods__)


def to_train_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = True
    try:
        m.set_phase(0)
    except:
        pass
    

def set_io_enable(m):
    try:
        m.set_enable()
    except:
        pass
    

def set_io_disable(m):
    try:
        m.set_disable()
    except:
        pass
    
    
def _clip_grad_norm_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


@register_method
def train_fn(self, **kwargs):
    self.model.apply(to_train_mode)
    self.model.train()
    
    parameter_manager = ParameterManager(**kwargs)

    liveplot = parameter_manager.get_param('liveplot', required = True)
    batch_size = parameter_manager.get_param('batch_size')
    epoch = parameter_manager.get_param('epoch')
    inner_epochs = parameter_manager.get_param('inner_epochs')
    warmup = parameter_manager.get_param('warmup')
    weight_decay = parameter_manager.get_param('weight_decay')
    weight_decay_bias = parameter_manager.get_param('weight_decay_bias')
    bias_scale = parameter_manager.get_param('bias_scale')
    accum_grad = parameter_manager.get_param('accum_grad')
    mixout = parameter_manager.get_param('mixout')
    ema = parameter_manager.get_param('ema')
    ema_freq = parameter_manager.get_param('ema_freq')
    ema_momentum = parameter_manager.get_param('ema_momentum')
    grad_clip = parameter_manager.get_param('grad_clip')
    rho = ema_momentum ** ema_freq

    loss_record = 0
    norm_record = 0
    metrics_record = {key.replace('*', ''): 0 for key in self.model.monitor_nodes}

    
    for i in self.optimizer:
        i.zero_grad()
    
    iteration_idx = 0
    while True:
        # Set LR
        cur_lr = self.annealer.get_trace('lr')
        update_optim(self.optimizer, [cur_lr, cur_lr * bias_scale, cur_lr, cur_lr], key = 'lr')

        # Set weight decay
        if weight_decay > 0:
            cur_weight_decay = self.annealer.get_trace('weight_decay')
            update_optim(
                self.optimizer, [cur_weight_decay, (cur_weight_decay / bias_scale) if weight_decay_bias else 0, 0, cur_weight_decay / 10],
                key = 'weight_decay'
            )
        
        # Set momentum
        cur_mom = self.annealer.get_trace('momentum')
        update_optim(self.optimizer, [cur_mom, cur_mom, cur_mom], key = 'momentum')
            
        # //////////////////////////
        # // Inner Loop
        # //////////////////////////
        
        for inner_loop in range(inner_epochs):
            # Inner loop shared the same mini-batch
            if inner_loop == 0:
                self.model.apply(set_io_enable)
            else:
                self.model.apply(set_io_disable)
            
            # 0: train, 1: val, 2: custom
            results = self.model({'phase': 0, 'batch_size': batch_size, 'inner_loop': inner_loop}) 
            
            loss = [results[i] for i in self.model.optimize_nodes]
            loss = sum(loss)
            loss.backward()

            self.accum_idx += 1
            self.accum_idx %= accum_grad

            if self.accum_idx == 0 or ((iteration_idx == self.train_loader_length) and (inner_loop == inner_epochs - 1)):
                for p in self.model.parameters():
                    try:
                        if p.requires_grad: p.grad.div_(accum_grad)
                    except:
                        pass

                if grad_clip != 0:
                    norm = _clip_grad_norm_(self.model.parameters(), grad_clip)
                    norm_record += norm

                for i in self.optimizer:
                    i.step()
                    i.zero_grad()

            loss_record += loss.item()
            
        # //////////////////////////
        # // End of Inner Loop
        # //////////////////////////
        
        if mixout > 0:
            _ratio = cur_lr * mixout
            for v, p_v, v_mask in zip(
                self.model.parameters(),
                self.mixout_prior.parameters(),
                self.mixout_mask
            ):
                mask = (v_mask.uniform_() < (_ratio)).to(v.dtype)
                v.data = mask * (_ratio * p_v.data + (1-_ratio) * v.data)  + (1-mask) * v.data
            
        
        if ema and (iteration_idx + 1) % ema_freq == 0:
            for v, ema_v in zip(
                self.model.state_dict().values(),
                self.val_model.state_dict().values()
            ):
                if ema_v.dtype == torch.int64:
                    # do not ema on int data for precision (underflow) issue
                    pass
                else:
                    ema_v.mul_(rho)
                    ema_v.add_(1-rho, v)

        for i in self.model.monitor_nodes:
            if '*' in i:
                continue
            metrics_record[i] += results[i].item()
        
        iteration_idx += 1
        
        if (self.train_loader_length > 1 and iteration_idx == self.train_loader_length - 1) or self.train_loader_length == 1:
            visualize = [results[i] for i in self.model.visualize_nodes]
        
        if liveplot is not None and (iteration_idx % 5 == 0 or iteration_idx == self.train_loader_length):
            liveplot.update_train_desc(epoch, iteration_idx, loss_record, metrics_record, self.val_model.checkpoint_state)

        if iteration_idx == self.train_loader_length:
            if liveplot is not None:
                liveplot.show_grid('train', visualize)
            break

        self.annealer._iteration_step()
    
    loss_record /= iteration_idx
    norm_record /= iteration_idx
    
    for i in metrics_record:
        metrics_record[i] /= iteration_idx
    
    return loss_record, norm_record, metrics_record