import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager
from ..ops import op
from ..models import Builder
import math


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
def validate(self, plot_wrong = False, out = 'categorical', **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    self.val_model.apply(_clear_source)
    self.val_model.apply(_clear_output)
    
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = 2
    
    for k, v in self.val_model.graph.items():
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
        loss_record, metrics_record = self.validate_fn(plot_wrong, **kwargs)
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
def validate_fn(self, plot_wrong = False, out = 'categorical', **kwargs):
    self.val_model.apply(to_eval_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    liveplot = parameter_manager.get_param('liveplot', None)
    epoch = parameter_manager.get_param('epoch', 1)
    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')

    self.val_model.eval()

    loss_record = 0
    metrics_record = {key.replace('*', ''): 0 for key in self.val_model.monitor_nodes}
    wrong_samples = []

    with torch.no_grad():
        batch_idx = 0
        while True:
            results = self.val_model({'phase': 1})
            
            if plot_wrong:
                wrong_samples.append(results[out])
            
            loss = sum([results[i] for i in self.model.optimize_nodes])

            for i in self.val_model.monitor_nodes:
                if '*' in i:
                    i = i.replace('*', '')
                metrics_record[i] += results[i]

            loss_record += loss.item()
            
            batch_idx += 1
            
            if (self.test_loader_length > 1 and batch_idx == self.test_loader_length - 1) or self.test_loader_length == 1:
                visualize = [results[i] for i in self.val_model.visualize_nodes]
            
            if liveplot and (batch_idx % 5 == 0 or batch_idx == self.test_loader_length):
                liveplot.update_val_desc(epoch, batch_idx, loss_record, metrics_record, self.val_model.checkpoint_state)
            
            if batch_idx == self.test_loader_length:
                if liveplot is not None:
                    liveplot.show_grid('val', visualize)
                break

    loss_record /= batch_idx
    
    for i in metrics_record:
        metrics_record[i] /= batch_idx
    
    print(wrong_samples)
    
    return loss_record, metrics_record
