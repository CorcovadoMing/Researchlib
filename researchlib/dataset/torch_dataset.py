import torchvision
import torch
import numpy as np
import random
from .np_dataset import _NumpyDataset


def _TorchDataset(name, is_train, shuffle, label_noise=0):
    dataset_fn = None
    for i in torchvision.datasets.__dict__:
        if i.lower() == name and type(torchvision.datasets.__dict__[i]) == type:
            dataset_fn = torchvision.datasets.__dict__[i]
            break

    if dataset_fn is None:
        raise ValueError(f'No dataset {name} founded')

    ds = dataset_fn(train = is_train, download = True, root = './data')
    
    if name == 'mnist':
        ds.data = ds.data.unsqueeze(-1)
    
    if type(ds.data) == torch.Tensor:
        data = ds.data.numpy().astype(np.float32)
        target = ds.targets.numpy() 
    else:
        data = np.array(ds.data).astype(np.float32)
        target = np.array(ds.targets)
    
    if label_noise != 0:
        noise_idx = random.choices(list(range(len(target))), k = int(len(target)*label_noise))
        target[noise_idx] = (target[noise_idx] + 1) % 10
        
    return _NumpyDataset(data, target, shuffle, name = name)