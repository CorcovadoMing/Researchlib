import torch.nn.init as init
from ..utils import _register_method
from torch import nn
import torch
from tqdm.auto import tqdm
import sys

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def init_model(
    self, init_algorithm = 'default', lsuv_dummy = False, lsuv_trials = 50, verbose = False
):
    
    # Reset the metrics
    if self.monitor_mode == min:
        self.monitor = 1e5
    elif self.monitor_mode == max:
        self.monitor = -1e5
    
    if init_algorithm == 'lsuv':
        init_distribution = 'orthogonal'
    else:
        init_distribution = init_algorithm

    hook = []

    def hook_fn(module, input, output):
        with torch.no_grad():
            for e in range(3):
                tmp = module._forward_hooks
                module._forward_hooks = {}
                out = module(input[0])
                out = out.std()
                module._forward_hooks = tmp
                for i in module.parameters():
                    if i.dim() > 1:
                        i /= out
    
    def _to_fp32(m):
        if isinstance(m, torch.nn.Module) and not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.float()

    def _init(m):
        if init_distribution == 'default':
            try:
                m.reset_parameters()
            except:
                pass
        else:
            if type(m) == nn.ModuleList or \
                        'researchlib.blocks' in str(type(m)) or \
                        'researchlib.models' in str(type(m)) or \
                        'researchlib.ops.condition_projection' in str(type(m)):
                pass
            else:
                for i in m.parameters():
                    if i.dim() > 1:
                        if verbose:
                            print('Initialize to ' + str(init_distribution) + ' :', m)
                        if init_distribution == 'xavier_normal':
                            init.xavier_normal_(i)
                        elif init_distribution == 'xavier_uniform':
                            init.xavier_uniform_(i)
                        elif init_distribution == 'orthogonal':
                            init.orthogonal_(i)
                        elif init_distribution == 'kaiming_uniform':
                            init.kaiming_uniform_(i)
                        elif init_distribution == 'kaiming_normal':
                            init.kaiming_normal_(i)

    def _lsuv(m):
        if type(m) == nn.ModuleList or \
                    'researchlib.blocks' in str(type(m)) or \
                    'researchlib.models' in str(type(m)) or \
                    'researchlib.ops.condition_projection' in str(type(m)):
            pass
        else:
            if len(list(m.parameters())) > 0:
                handler = m.register_forward_hook(hook_fn)
                hook.append(handler)

    self.model.apply(_to_fp32)
    self.model.apply(_init)

    if init_algorithm == 'lsuv':
        self.model.apply(_lsuv)
        try:
            self.preload_gpu()
            bar = tqdm(range(lsuv_trials), leave = False, initial = 1)
            for i, data_pack in enumerate(self.train_loader):
                if lsuv_dummy:
                    dummy_pack = [
                        torch.Tensor(i.size()).uniform_().to(i.device) for i in data_pack
                    ]
                    self.fit_xy(dummy_pack, 1, _train = False)
                else:
                    self.fit_xy(data_pack, 1, _train = False)
                bar.update(1)
                if i == lsuv_trials - 1:
                    bar.close()
                    break
        except:
            raise
        finally:
            self.unload_gpu()
            for i in hook:
                i.remove()
