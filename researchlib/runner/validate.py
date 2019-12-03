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
def validate(self, monitor = [], visualize = [], prefetch = True, **kwargs):
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
        loss_record, metrics_record, visualize_record = self.validate_fn(monitor, visualize, **kwargs)
        print(loss_record)
        for k, v in metrics_record.items():
            print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.val_model.apply(_clear_source)
        self.val_model.apply(_clear_output)
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
