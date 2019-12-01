import torch
from ..utils import _register_method, ParameterManager
from ..ops import op
from ..models import Builder
from torch import nn, optim
from torch.autograd import Variable


__methods__ = []
register_method = _register_method(__methods__)


def to_eval_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = False
    try:
        m.set_phase(1)
    except:
        pass


@register_method
def calibrate(self, monitor = [], visualize = [], prefetch = True, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    fp16 = parameter_manager.get_param('fp16', False)
    buffered_epochs = 1
    
    self.preload_gpu()
    try:
        loss_record, metrics_record, visualize_record = self.calibrate_fn(monitor, visualize, **kwargs)
        print(loss_record)
        for k, v in metrics_record.items():
            print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.unload_gpu()


@register_method
def calibrate_fn(self, monitor, visualize, **kwargs):
    self.val_model.apply(to_eval_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    self.val_model.eval()
    
    calibrate_node = self.val_model.graph[self.output_node][0]
    calibrate_node.temperature.requires_grad = True
#     calibrate_loss = self.val_model.graph[self.loss_fn][0]
#     calibrate_loss.train()
    optimizer = optim.SGD(calibrate_node.parameters(), lr=1)

    loss_record = 0
    metrics_record = {key: 0 for key in monitor}
    visualize_record = {key: 0 for key in visualize}

    batch_idx = 0
    while True:
        results = self.val_model({'phase': 1})

        optimizer.zero_grad()
        print(calibrate_node.temperature)
        print(results[self.loss_fn])
        loss = results[self.loss_fn]
        print(calibrate_node.temperature.grad)
        loss.backward()
        print(calibrate_node.temperature.grad)
        optimizer.step()


        batch_idx += 1
        if batch_idx == self.test_loader_length:
            break

    loss_record /= batch_idx
    
    for i in metrics_record:
        metrics_record[i] /= batch_idx
    
    return loss_record, metrics_record, visualize_record
