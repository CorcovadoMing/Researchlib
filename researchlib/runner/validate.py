import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager, inifinity_loop
from functools import reduce
from .prefetch import BackgroundGenerator

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def validate(self, metrics = [], callbacks = [], prefetch=True, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    
    buffered_epochs = 100
    test_loader = self.test_loader.get_generator(batch_size, epochs=buffered_epochs)
    self.test_loader_length = len(test_loader)
    test_loader = BackgroundGenerator(inifinity_loop(test_loader))
    self.preload_gpu()
    try:
        if len(self.default_metrics):
            metrics = self.default_metrics + metrics

        loss_record = self.validate_fn(test_loader, metrics)

        print(loss_record)

        if len(metrics) > 0:
            for m in metrics:
                for k, v in m.output().items():
                    print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.unload_gpu()
            
            
@register_method
def validate_fn(self, loader, metrics):
    self.model.eval()

    for m in metrics:
        m.reset()

    loss_record = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            outputs = self.model(inputs)
            loss = self.loss_fn[0](outputs, targets)

            for m in metrics:
                m.forward([outputs, targets])

            loss_record += loss.item()
            del loss

            if batch_idx == (self.test_loader_length-1):
                break

    loss_record = loss_record / (batch_idx + 1)
    return loss_record