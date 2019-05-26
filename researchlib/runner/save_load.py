import torch
from torch import nn
import warnings

def _save_model(model, path):
    path = path + '.model.h5'
    warnings.filterwarnings('ignore')    
    if type(model) == nn.DataParallel:
        torch.save(model.module, path)
    else:
        torch.save(model, path)
    warnings.filterwarnings('once')

def _save_optimizer(optimizer, path):
    path = path + '.optimizer.h5'
    warnings.filterwarnings('ignore')
    torch.save(optimizer.state_dict(), path)
    warnings.filterwarnings('once')

def _load_model(model, path, multigpu):
    path = path + '.model.h5'
    model = torch.load(path)
    if multigpu:
        model = nn.DataParallel(model)
    return model

def _load_optimizer(optimizer, path):
    path = path + '.optimizer.h5'
    optimizer.load_state_dict(torch.load(path))