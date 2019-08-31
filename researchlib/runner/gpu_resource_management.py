from ..utils import _register_method, _get_iteration, set_lr, plot_montage
import torch

__methods__ = []
register_method = _register_method(__methods__)


def _load_state(state, device):
    for key in state:
        for attr in state[key]:
            try:
                if device == 'gpu':
                    state[key][attr] = state[key][attr].cuda()
                elif device == 'cpu':
                    state[key][attr] = state[key][attr].cpu()
            except:
                pass


def _switch_optimizer_state(optim, device, lookahead):
    if lookahead:
        fast_state = optim.state_dict()['fast_state']
        slow_state = optim.state_dict()['slow_state']
        _load_state(fast_state, device)
        _load_state(slow_state, device)
    else:
        try:
            state = optim.state_dict()['state']
        except:
            state = optim.state_dict()['opt_state']
        _load_state(state, device)


@register_method
def preload_gpu(self):
    if self.is_cuda:
        if self.optimizer is not None:
            if type(self.optimizer) == tuple or type(self.optimizer) == list:
                for optim in self.optimizer:
                    _switch_optimizer_state(optim, 'gpu', self.lookahead)
            else:
                _switch_optimizer_state(self.optimizer, 'gpu', self.lookahead)
        self.model.cuda()


@register_method
def unload_gpu(self):
    if self.is_cuda:
        if self.optimizer is not None:
            if type(self.optimizer) == tuple or type(self.optimizer) == list:
                for optim in self.optimizer:
                    _switch_optimizer_state(optim, 'cpu', self.lookahead)
            else:
                _switch_optimizer_state(self.optimizer, 'cpu', self.lookahead)
        self.model.cpu()
        torch.cuda.empty_cache()
