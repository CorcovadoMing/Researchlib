import torch
from ..utils import _register_method, get_aux_out, _switch_swa_mode, ParameterManager
from ..ops import op
from ..models import Builder


__methods__ = []
register_method = _register_method(__methods__)


__methods__ = []
register_method = _register_method(__methods__)


def to_predict_mode(m):
    if isinstance(m, Builder.Graph):
        m.train_mode = False
    try:
        m.set_phase(2)
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
def predict(self, dict_input, outputs=None, **kwargs):
    parameter_manager = ParameterManager(**kwargs)
    
    try:
        self.val_model
    except:
        self.val_model = self.model
        
    self.val_model.apply(_clear_source)
    self.val_model.apply(_clear_output)
    
    self.preload_gpu()
    try:
        results = self.predict_fn(dict_input, outputs, **kwargs)
    except:
        raise
    finally:
        self.val_model.apply(_clear_source)
        self.val_model.apply(_clear_output)
        self.unload_gpu()
    return results


@register_method
def predict_fn(self, dict_input, outputs, **kwargs):
    self.val_model.apply(to_predict_mode)
    
    parameter_manager = ParameterManager(**kwargs)

    support_set = parameter_manager.get_param('support_set')
    way = parameter_manager.get_param('way')
    shot = parameter_manager.get_param('shot')

    self.val_model.eval()

    with torch.no_grad():
        dict_input['phase'] = 2
        # TODO:
        # 1. GPU
        # 2. FP16
        # 3. Normalizer
        # 4. Tensors, etc.,
        if outputs is None:
            # Forward the whole compute graph
            return self.val_model(dict_input)
        else:
            # TODO: if all the outputs is collected, return immediately
            return 0
    