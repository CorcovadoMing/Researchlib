import torchvision
import torch
from .np_dataset import _NumpyDataset


def _TorchDataset(name, is_train, shuffle):
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
        return _NumpyDataset(ds.data.numpy(), ds.targets.numpy(), shuffle, name = name)
    else:
        return _NumpyDataset(ds.data, ds.targets, shuffle, name = name)