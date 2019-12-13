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
import numpy as np

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
def attack(self, epsilon = 0.01, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    self.val_model.apply(_clear_source)
    self.val_model.apply(_clear_output)
    
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
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
        self.attack_fn(epsilon, **kwargs)
    except:
        raise
    finally:
        self.val_model.apply(_clear_source)
        self.val_model.apply(_clear_output)
        self.unload_gpu()


def _enable_grad(var):
    var.requires_grad = True
    return var

import matplotlib.pyplot as plt
class _FGSM:
    def __init__(self, attack_gradient, epsilon):
        super().__init__()
        self.cur_idx = 0
        self.attack_gradient = attack_gradient
        self.epsilon = epsilon
    
    def __call__(self, x):
        with torch.no_grad():
            data_grad = self.attack_gradient[self.cur_idx].sign() * 255
            data_grad = data_grad.detach().cpu().numpy().transpose(0, 2, 3, 1)
            x = x + self.epsilon * data_grad
            np.clip(x, 0, 255, x)
            self.cur_idx += 1
            self.cur_idx %= len(self.attack_gradient)
            return x

@register_method
def attack_fn(self, epsilon, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = 2
    
    self.val_model.apply(to_eval_mode)
    self.val_model.eval()
    
    # 1. Get the attack target and store the attack gradient
    Callback.inject_after(self.val_model.graph, 'x', _enable_grad)
    
    attack_gradient = []
    batch_idx = 0
    pbar = tqdm(total=self.test_loader_length)
    count = 0
    while True:
        results = self.val_model({'phase': 1})
        loss = [results[i] for i in self.model.optimize_nodes]
        loss = sum(loss)
        self.val_model.zero_grad()
        loss.backward()
        count += results['acc'].item()
        attack_gradient.append(results['x'].grad.data)
        batch_idx += 1
        pbar.update(1)
        if batch_idx == self.test_loader_length:
            break
            
    pbar.close()
    print(count / batch_idx)
    Callback.restore_inject(self.val_model.graph, 'x')
    
    
    # 2. Reference on attack images
    self.val_model.apply(_clear_source)
    self.val_model.apply(_clear_output)
    
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
            
    self.val_model.graph['normalize'][0].set_injector(_FGSM(attack_gradient, epsilon))
    batch_idx = 0
    pbar = tqdm(total=self.test_loader_length)
    count = 0
    while True:
        results = self.val_model({'phase': 1})
        count += results['acc'].item()
        batch_idx += 1
        pbar.update(1)
        if batch_idx == self.test_loader_length:
            break
            
    pbar.close()
    print(count / batch_idx)
    self.val_model.graph['normalize'][0].clear_injector()
    