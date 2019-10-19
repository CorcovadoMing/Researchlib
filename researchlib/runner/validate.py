import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager, inifinity_loop
from functools import reduce
from .prefetch import BackgroundGenerator

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def validate(self, metrics = None, monitor = [], prefetch = True, **kwargs):
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
        loss_record, metrics_record = self.validate_fn(test_loader, metrics, monitor)
        print(loss_record)
        for k, v in metrics_record.items():
            print(str(k) + ':', float(v))
    except:
        raise
    finally:
        self.unload_gpu()


@register_method
def validate_fn(self, loader, metrics, monitor, **kwargs):
    parameter_manager = ParameterManager(**kwargs)

    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')

    self.val_model.eval()

    loss_record = 0
    metrics_record = {key: 0 for key in monitor}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if support_set is not None:
                support_x, support_y = support_set
                support_x = torch.from_numpy(support_x).to(inputs.device).to(inputs.dtype)
                support_x = support_x.view(-1, *inputs.shape[1:])
                outputs = self.val_model(inputs, support_x)  # Batch, shot, way

                # Deal with labels
                support_y = torch.from_numpy(support_y).to(targets.device).to(targets.dtype)
                support_y = support_y.expand(targets.size(0), -1,
                                             -1).transpose(-1, -2)  # Batch, shot, way
                targets = targets.unsqueeze(1).unsqueeze(2).expand(-1, shot, len(way))
                targets = targets.eq(support_y).to(inputs.dtype)
            else:
                #outputs = self.val_model(inputs)
                outputs = (self.val_model(inputs) + self.val_model(torch.flip(inputs, [-1]))) / 2

            loss = self.loss_fn[0](outputs, targets)

            metrics_result = metrics({
                'x': outputs,
                'y': targets
            }) if metrics is not None else metrics_record
            for i in monitor:
                metrics_record[i] += metrics_result[i]

            loss_record += loss.item()
            del loss

            if batch_idx == (self.test_loader_length - 1):
                break

    loss_record = loss_record / (batch_idx + 1)
    for i in metrics_record:
        metrics_record[i] /= (batch_idx + 1)
    return loss_record, metrics_record
