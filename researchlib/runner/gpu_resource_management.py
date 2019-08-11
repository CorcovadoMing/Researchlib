from ..utils import _register_method, _get_iteration, set_lr, plot_montage
import torch

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def preload_gpu(self):
    if self.is_cuda:
        if type(self.optimizer) == tuple or type(self.optimizer) == list:
            for optim in self.optimizer:
                try:
                    state = self.optimizer.state_dict()['state']
                except:
                    state = self.optimizer.state_dict()['opt_state']

                for key in state:
                    for attr in state[key]:
                        try:
                            state[key][attr] = state[key][attr].cuda()
                        except:
                            pass
        else:
            try:
                state = self.optimizer.state_dict()['state']
            except:
                state = self.optimizer.state_dict()['opt_state']

            for key in state:
                for attr in state[key]:
                    try:
                        state[key][attr] = state[key][attr].cuda()
                    except:
                        pass
        self.model.cuda()


@register_method
def unload_gpu(self):
    if self.is_cuda:
        if type(self.optimizer) == tuple or type(self.optimizer) == list:
            for optim in self.optimizer:
                try:
                    state = self.optimizer.state_dict()['state']
                except:
                    state = self.optimizer.state_dict()['opt_state']

                for key in state:
                    for attr in state[key]:
                        try:
                            state[key][attr] = state[key][attr].cuda()
                        except:
                            pass
        else:
            try:
                state = self.optimizer.state_dict()['state']
            except:
                state = self.optimizer.state_dict()['opt_state']

            for key in state:
                for attr in state[key]:
                    try:
                        state[key][attr] = state[key][attr].cuda()
                    except:
                        pass
        self.model.cpu()
        torch.cuda.empty_cache()
