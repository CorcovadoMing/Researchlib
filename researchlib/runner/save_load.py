import torch
from torch import nn
import warnings

def _save_model(model, path):
    warnings.filterwarnings('ignore')    
    if type(model) == nn.DataParallel:
        torch.save(model.module, path)
    else:
        torch.save(model, path)
    warnings.filterwarnings('once')
    
def _load_model(model, path, multigpu):
    model = torch.load(path)
    if multigpu:
        model = nn.DataParallel(model)
    return model