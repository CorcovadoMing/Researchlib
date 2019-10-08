import torch
from torch import nn
import warnings
from apex import amp

def _save_checkpoint(model, optimizer, path):
    path = path + '.model.pt'
    warnings.filterwarnings('ignore')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        #'amp': amp.state_dict()
    }
    torch.save(checkpoint, path)
    warnings.filterwarnings('once')


def _load_checkpoint(model, optimizer, multigpu, path):
    path = path + '.model.pt'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # TODO: may need to load checkpoint['amp']
    if multigpu:
        model = nn.DataParallel(model)
    return model, optimizer
