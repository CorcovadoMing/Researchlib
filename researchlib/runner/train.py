import torch
from ..utils import _register_method, ParameterManager, Annealer, set_lr, update_optim
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

        
@register_method
def train_fn(self, monitor, visualize, **kwargs):
    self.model.apply(to_train_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    liveplot = parameter_manager.get_param('liveplot', required = True)
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

    loss_record = 0
    norm_record = 0
    metrics_record = {key: 0 for key in monitor}
    visualize_record = {key: 0 for key in visualize}

    batch_idx = 0
    while True:
        # Set LR
        cur_lr = Annealer.get_trace('lr')
        update_optim(self.optimizer, [cur_lr, cur_lr * bias_scale], key = 'lr')

        # Set weight decay
        if weight_decay > 0:
            cur_weight_decay = Annealer.get_trace('weight_decay')
            update_optim(
                self.optimizer, [cur_weight_decay, cur_weight_decay / bias_scale],
                key = 'weight_decay'
            )

        if mmixup_alpha is not None:
            batch_size = inputs[0].size(0)
            if fixed_mmixup is None and random_mmixup is None:
                random_mmixup = [0, op.ManifoldMixup.block_counter]
            lam = op.ManifoldMixup.setup_batch(
                mmixup_alpha, batch_size, fixed_mmixup, random_mmixup
            )
            targets, targets_res = op.ManifoldMixup.get_y(targets)
        else:
            targets_res = None
            lam = None

        for i in self.optimizer:
            i.zero_grad()
        
        results = self.model({'phase': 0}) # 0: train, 1: val, 2: custom
        loss = results[self.loss_fn]
        loss.backward()

        for i in self.optimizer:
            i.step()

        loss_record += loss.item()
        del loss

        if ema and (batch_idx + 1) % ema_freq == 0:
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

        for i in monitor:
            metrics_record[i] += results[i]
        
        for i in visualize:
            visualize_record[i] += results[i]
        
        del results

        
        batch_idx += 1
        if batch_idx % 5 == 0 or batch_idx == self.train_loader_length:
            liveplot.update_desc(epoch, batch_idx, loss_record, metrics_record, self.monitor)

        if batch_idx == self.train_loader_length:
            break

        Annealer._iteration_step()

    loss_record /= batch_idx
    norm_record = (norm_record ** 0.5) / batch_idx
    
    for i in metrics_record:
        metrics_record[i] /= batch_idx
    
    return loss_record, norm_record, metrics_record, visualize_record



#         if support_set is not None:
#             support_x, support_y = support_set
#             support_x = torch.from_numpy(support_x).to(inputs.device).to(inputs.dtype)
#             support_x = support_x.view(-1, *inputs.shape[1:])
#             outputs = self.model(inputs, support_x)  # Batch, shot, way

#             # Deal with labels
#             support_y = torch.from_numpy(support_y).to(targets.device).to(targets.dtype)
#             support_y = support_y.expand(targets.size(0), -1, -1).transpose(-1, -2)  # Batch, shot, way
#             targets = targets.unsqueeze(1).unsqueeze(2).expand(-1, shot, len(way))
#             targets = targets.eq(support_y).to(inputs.dtype)
#             targets *= (1 - 0.2 - (0.2/9))
#             targets += (0.2/9)
#         else:
#             support_x, support_y = None, None