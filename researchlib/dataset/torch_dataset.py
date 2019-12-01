import torchvision
from .tp_dataset import _NumpyDataset


def _TorchDataset(name, is_train, shuffle):
    dataset_fn = None
    for i in torchvision.datasets.__dict__:
        if i.lower() == name:
            dataset_fn = torchvision.datasets.__dict__[i]
            break

    if dataset_fn is None:
        raise ValueError(f'No dataset {name} founded')

    ds = dataset_fn(train = is_train, download = True, root = './data')
    return _NumpyDataset(ds.data, ds.targets, shuffle, name = name)