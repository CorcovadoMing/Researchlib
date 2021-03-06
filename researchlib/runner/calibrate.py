import torch
from torch import nn
from ..utils import _register_method, ParameterManager
from ..ops import op
from ..models import Builder
from torch import nn, optim
from torch.autograd import Variable
from ..metrics import Metrics

__methods__ = []
register_method = _register_method(__methods__)

import matplotlib.pyplot as plt
def make_model_diagrams(outputs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels).float()

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_corrects = torch.stack([torch.mean(accuracies[bin_index]).view(-1) if len(accuracies[bin_index]) else torch.zeros(1) for bin_index in bin_indices]).view(-1)
    bin_scores = torch.stack([torch.mean(confidences[bin_index]).view(-1) if len(confidences[bin_index]) else torch.zeros(1) for bin_index in bin_indices]).view(-1)

    print(bin_corrects)
    print(bin_scores)
    print(bin_scores - bin_corrects)
    
    plt.plot(bin_corrects.detach().numpy())
    plt.plot(bin_scores.detach().numpy())
    plt.show()


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
def calibrate(self, update=[], **kwargs):
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
        self.calibrate_fn(update, **kwargs)
    except:
        raise
    finally:
        self.val_model.apply(_clear_source)
        self.val_model.apply(_clear_output)
        self.unload_gpu()


@register_method
def calibrate_fn(self, update, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    temperature = Variable(torch.ones(1), requires_grad=True)
    
    def temperature_scale(logits):
        return logits / temperature
    
    self.val_model.apply(to_eval_mode)
    self.val_model.eval()
    
    ece_metrics = Metrics.ECE()
    nll = nn.CrossEntropyLoss()
    
    batch_idx = 0
    logits = []
    labels = []
    while True:
        results = self.val_model({'phase': 1})
        
        # TODO: Several output nodes should calibrate individually
        logits.append(results[self.val_model.output_nodes[0]].detach().cpu())
        labels.append(results['y'].detach().cpu())
        
        batch_idx += 1
        if batch_idx == self.test_loader_length:
            break

    logits = torch.cat(logits).float()
    labels = torch.cat(labels).long()
    
    before_ece = ece_metrics(logits, labels)
    before_nll = nll(logits, labels)
    
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))
    make_model_diagrams(logits, labels)
    
    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        loss = nll(temperature_scale(logits), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_nll = nll(temperature_scale(logits), labels)
    after_ece = ece_metrics(temperature_scale(logits), labels)
    
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_nll, after_ece))
    print('Optimal temperature: %.3f' % temperature)
    
    make_model_diagrams(temperature_scale(logits), labels)

    for i in update:
        self.val_model.graph[i][0].temperature.fill_(float(temperature))