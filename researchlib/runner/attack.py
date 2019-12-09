import torch
from torch import nn
from ..utils import _register_method, ParameterManager
from ..ops import op
from ..models import Builder
from torch import nn, optim
from torch.autograd import Variable
from ..metrics import Metrics
from ..callback import Callback
from tqdm.auto import tqdm

__methods__ = []
register_method = _register_method(__methods__)

def to_eval_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = False
    try:
        m.set_phase(1)
    except:
        pass

def _clear_output(m):
    try:
        del m.outputs
    except:
        pass

def _clear_source(m):
    try:
        m.clear_source(False)
    except:
        pass


@register_method
def attack(self, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    self.val_model.apply(_clear_source)
    self.val_model.apply(_clear_output)
    
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = 1
    buffered_epochs = 2
    
    for k, v in self.val_model.graph.items():
        if type(v[0]) == op.Source:
            v[0].prepare_generator(batch_size, buffered_epochs)
            self.train_loader_length = v[0].train_source_generator.__len__()
            if v[0].val_source is not None:
                self.test_loader_length = v[0].val_source_generator.__len__()
            else:
                self.test_loader_length = None
        if type(v[0]) == op.Generator:
            v[0].prepare_state(fp16)
            
    self.preload_gpu()
    try:
        self.attack_fn(**kwargs)
    except:
        raise
    finally:
        self.val_model.apply(_clear_source)
        self.val_model.apply(_clear_output)
        self.unload_gpu()


def _enable_grad(var):
    var.requires_grad = True
    return var

class _FGSM(nn.Module):
    def __init__(self, attack_idx, attack_gradient, epsilon = 0.3):
        super().__init__()
        self.cur_idx = 0
        self.count = 0
        self.attack_idx = attack_idx
        self.attack_gradient = attack_gradient
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.cur_idx in self.attack_idx:
            data_grad = self.attack_gradient[self.count].sign()
            x = x + self.epsilon * data_grad
            x = torch.clamp(x, 0, 1) # This line may be check if the original range is not [0, 1]
            self.count += 1
        self.cur_idx += 1
        return x

@register_method
def attack_fn(self, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    # 1. Get the attack target and store the attack gradient
    Callback.inject_after(self.val_model.graph, 'x', _enable_grad)
    
    attack_idx = []
    attack_gradient = []
    batch_idx = 0
    pbar = tqdm(total=self.test_loader_length)
    while True:
        results = self.val_model({'phase': 1})
        if results['acc'] != 0:
            loss = [results[i] for i in self.model.optimize_nodes]
            loss = sum(loss)
            self.val_model.zero_grad()
            loss.backward()
            attack_idx.append(batch_idx)
            attack_gradient.append(results['x'].grad.data)
            
        batch_idx += 1
        pbar.update(1)
        if batch_idx == self.test_loader_length:
            break
            
    pbar.close()
    print(len(attack_idx))
    Callback.restore_inject(self.val_model.graph, 'x')
    
    
    # 2. Reference on attack images
    Callback.inject_after(self.val_model.graph, 'x', _FGSM(attack_idx, attack_gradient))
    batch_idx = 0
    pbar = tqdm(total=self.test_loader_length)
    count = 0
    while True:
        results = self.val_model({'phase': 1})
        count += results['acc']
        batch_idx += 1
        pbar.update(1)
        if batch_idx == self.test_loader_length:
            break
            
    pbar.close()
    print(count)
    Callback.restore_inject(self.val_model.graph, 'x')
    