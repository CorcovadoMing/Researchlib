import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager
from ..ops import op
from ..models import Builder


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
def validate(self, monitor = [], visualize = [], prefetch = True, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    fp16 = parameter_manager.get_param('fp16', False)
    buffered_epochs = 1
    
    self.preload_gpu()
    try:
        loss_record, metrics_record, visualize_record = self.validate_fn(monitor, visualize, **kwargs)
        print(loss_record)
        for k, v in metrics_record.items():
            print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.unload_gpu()


@register_method
def validate_fn(self, monitor, visualize, **kwargs):
    self.val_model.apply(to_eval_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')

    self.val_model.eval()

    loss_record = 0
    metrics_record = {key: 0 for key in monitor}
    visualize_record = {key: 0 for key in visualize}

    with torch.no_grad():
        batch_idx = 0
        while True:
            results = self.val_model({'phase': 1})
            loss = results[self.loss_fn]

            for i in monitor:
                metrics_record[i] += results[i]

            for i in visualize:
                visualize_record[i] += results[i]
                
            del results
                
            loss_record += loss.item()
            del loss
            
            batch_idx += 1
            if batch_idx == self.test_loader_length:
                break

    loss_record /= batch_idx
    
    for i in metrics_record:
        metrics_record[i] /= batch_idx
    
    return loss_record, metrics_record, visualize_record
