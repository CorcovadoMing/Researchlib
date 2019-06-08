import torch.nn.init as init
from ..utils import _register_method
from torch import nn
import torch
from tqdm.auto import tqdm

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def init_model(self, init_distribution='xavier_normal', verbose=False):
    def _init(m):
        if type(m) == nn.ModuleList or \
                    'researchlib.layers.block' in str(type(m)) or \
                    'researchlib.models' in str(type(m)) or \
                    'researchlib.layers.condition_projection' in str(type(m)):
            pass
        else:
            for i in m.parameters():
                if i.dim() > 1:
                    if verbose:
                        print('Initialize to ' + str(init_distribution) + ' :', type(m), m)
                    if init_distribution == 'xavier_normal':
                        init.xavier_normal_(i)
                    elif init_distribution == 'orthogonal':
                        init.orthogonal_(i)
                else:
                    init.normal_(i)
    self.model.apply(_init)
                

@register_method
def test_init_model(self, init_distribution='xavier_normal', verbose=False):
    hook = []
    def hook_fn(module, input, output):
        with torch.no_grad():
            if type(output) == tuple or type(output) == list:
                out = [i.std for i in output].sum()
            else:
                out = output.std()

            if torch.abs(out - 1) > 0.1: 
                for i in module.parameters():
                    if i.dim() > 1 and out.log() > 1e-2:
                        i /= out.log()

        
    def _init(m):
        if type(m) == nn.ModuleList or \
                    'researchlib.layers.block' in str(type(m)) or \
                    'researchlib.models' in str(type(m)) or \
                    'researchlib.layers.condition_projection' in str(type(m)):
            pass
        else:
            if len(list(m.parameters())) > 0:
                handler = m.register_forward_hook(hook_fn)
                hook.append(handler)
    self.model.apply(_init)
    try:
        self.preload_gpu()
        trials = 200
        bar = tqdm(range(trials), leave=False, initial=1)
        for i, data_pack in enumerate(self.train_loader):
            self.fit_xy(data_pack, 1, _train=False)
            bar.update(1)
            if i > trials: 
                bar.close()
                break
    except:
        raise
    finally:
        self.unload_gpu()
        for i in hook:
            i.remove()