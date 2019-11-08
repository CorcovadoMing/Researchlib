import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager, inifinity_loop
from functools import reduce
from .prefetch import BackgroundGenerator
from ..models import Builder


__methods__ = []
register_method = _register_method(__methods__)


def to_eval_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = False
        
        
@register_method
def validate(self, monitor = [], visualize = [], prefetch = True, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    batch_size = parameter_manager.get_param(
        'batch_size', 512, validator = lambda x: x > 0 and type(x) == int
    )
    
    fp16 = parameter_manager.get_param('fp16', False)

    buffered_epochs = 1
    test_loader = self.test_loader.get_generator(batch_size, epochs = buffered_epochs)
    self.test_loader_length = len(test_loader)
    test_loader = BackgroundGenerator(inifinity_loop(test_loader), fp16 = fp16)
    self.preload_gpu()
    try:
        loss_record, metrics_record, visualize_record = self.validate_fn(
            test_loader, monitor, visualize, **kwargs
        )
        print(loss_record)
        for k, v in metrics_record.items():
            print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.unload_gpu()


@register_method
def validate_fn(self, loader, monitor, visualize, **kwargs):
    self.val_model.apply(to_eval_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')
    tta = parameter_manager.get_param('tta', False)

    self.val_model.eval()

    loss_record = 0
    metrics_record = {key: 0 for key in monitor}
    visualize_record = {key: 0 for key in visualize}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if support_set is not None:
                support_x, support_y = support_set
                support_x = torch.from_numpy(support_x).to(inputs.device).to(inputs.dtype)
                support_x = support_x.view(-1, *inputs.shape[1:])
                outputs = self.model(inputs, support_x)  # Batch, shot, way

                # Deal with labels
                support_y = torch.from_numpy(support_y).to(targets.device).to(targets.dtype)
                support_y = support_y.expand(targets.size(0), -1, -1).transpose(-1, -2)  # Batch, shot, way
                targets = targets.unsqueeze(1).unsqueeze(2).expand(-1, shot, len(way))
                targets = targets.eq(support_y).to(inputs.dtype)
            else:
                support_x, support_y = None, None

            results = self.val_model({
                'x': inputs, 
                'y': targets, 
                'support_x': support_x, 
                'support_y': support_y
            })

            if tta:
                results_tta = self.val_model({
                    'x': torch.flip(inputs, [-1]), 
                    'y': targets, 
                    'support_x': support_x, 
                    'support_y': support_y
                })
                outputs = (results[self.output_node] + results_tta[self.output_node]) / 2
                loss = (results[self.loss_fn] + results_tta[self.loss_fn]) / 2
            else:
                outputs = results[self.output_node]
                loss = results[self.loss_fn]
                

            for i in monitor:
                metrics_record[i] += results[i]

            for i in visualize:
                visualize_record[i] += results[i]
                
            loss_record += loss.item()
            del loss

            if batch_idx == (self.test_loader_length - 1):
                break

    loss_record = loss_record / (batch_idx + 1)
    
    for i in metrics_record:
        metrics_record[i] /= (batch_idx + 1)
    
    return loss_record, metrics_record, visualize_record
