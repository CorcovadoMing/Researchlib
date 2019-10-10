import torch
from torch.optim.optimizer import Optimizer

class NAG(Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                w = p.data
                dw = p.grad.data
                
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(w)
                    
                v = param_state['momentum_buffer']

                dw.add_(weight_decay, w).mul_(-lr)
                v.mul_(momentum).add_(dw)
                w.add_(dw.add_(momentum, v))

        return loss