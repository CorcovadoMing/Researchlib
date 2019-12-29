import torch
from torch import nn
from ..utils import ParameterManager
import warnings


def _save_checkpoint(model, optimizer, path):
    model_path = path + '.model.pt'
    params_path = path + '.param.pt'
    
    # Save model/optimizer
    warnings.filterwarnings('ignore')
    
    if len(optimizer) == 2:
        optimizer_bias = optimizer[1].state_dict()
    else:
        optimizer_bias = None
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer_weight': optimizer[0].state_dict(),
        'optimizer_bias': optimizer_bias,
    }
    torch.save(checkpoint, model_path)
    
    # Save parameter manager state
    ParameterManager.dump(params_path)
    warnings.filterwarnings('once')


def _load_checkpoint(model, optimizer, multigpu, path):
    model_path = path + '.model.pt'
    params_path = path + '.param.pt'
    
    # Load model/optimizer
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer[0].load_state_dict(checkpoint['optimizer_weight'])
    if len(optimizer) == 2:
        optimizer[1].load_state_dict(checkpoint['optimizer_bias'])
    if multigpu:
        model = nn.DataParallel(model)
        
    # Load parameter manager state
    ParameterManager.load(params_path)
    return model, optimizer
