import torch
from ..utils import _register_method, ParameterManager, Annealer, update_optim
from ..ops import op
from ..models import Builder
import math
from ..loss import Loss


__methods__ = []
register_method = _register_method(__methods__)


def to_train_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = True
    try:
        m.set_phase(0)
    except:
        pass

def _clear_source(m):
    try:
        m.clear_source(True)
    except:
        pass
    
    
def _to_half(m):
    if isinstance(m, torch.nn.Module) and not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.half()
            
def _fix(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
    if type(m) == Loss.AdaptiveRobust:
        m.float()


@register_method
def step(self, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = 2
    
    self.model.apply(_clear_source)
    
    for k, v in self.model.graph.items():
        if type(v[0]) == op.Source:
            v[0].prepare_generator(buffered_epochs)
            self.train_loader_length = math.ceil(v[0].train_source_generator.__len__() / batch_size)
            if v[0].val_source is not None:
                self.test_loader_length = math.ceil(v[0].val_source_generator.__len__() / batch_size)
            else:
                self.test_loader_length = None
        if type(v[0]) == op.Generator:
            v[0].prepare_state(fp16, batch_size)
    
    self.preload_gpu()
    try:
        self.model.apply(to_train_mode)
        self.model.train()
        if fp16:
            self.model.apply(_to_half)
            self.model.apply(_fix)
        results = self.model({'phase': 0}) # 0: train, 1: val, 2: custom
    except:
        raise
    finally:
        self.unload_gpu()
    
    return results