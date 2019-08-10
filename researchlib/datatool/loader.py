import torch
import torch.utils.data as utils
import zarr
import os


class _ZarrDataset(utils.Dataset):
    def __init__(self, name):
        root = zarr.open(os.path.join(name, 'db.zarr'), mode='r')
        self.data = root.data
        self.label = root.label

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]), self.label[index])

    def __len__(self):
        return len(self.data)


class _Loader:
    def __init__(self):
        pass

    def project(self,
                name,
                inputs=1,
                batch_size=1,
                shuffle=True,
                sampler=None,
                batch_sampler=None,
                num_workers=2,
                drop_last=False):

        _dataset = _ZarrDataset(name)
        _dataset_loader = utils.DataLoader(_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           pin_memory=False,
                                           drop_last=drop_last)
        return _dataset_loader, inputs
