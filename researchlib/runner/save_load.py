import torch
from torch import nn

def _save_model(model, path):
    if type(model) == nn.DataParallel:
        torch.save(model.module, path)
    else:
        torch.save(model, path)

def _load_model(model, path, multigpu):
    model = torch.load(path)
    if multigpu:
        model = nn.DataParallel(model)
    return model