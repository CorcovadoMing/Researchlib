from torch.utils.data import Dataset, DataLoader
import torch, h5py

class _HDF5Dataset(Dataset):
    def __init__(self, path, key):
        if type(path) == list or type(path) == tuple:
            assert(len(path) == len(key))
            self.path = path
            self.key = key
        else:
            self.path = [path]
            self.key = [key]
        self.length = len(h5py.File(self.path[0], 'r')[self.key[0]])
    
    def __getitem__(self, index):
        return tuple(torch.from_numpy(h5py.File(i, 'r')[j][index]).float() for i, j in zip(self.path, self.key))
        
    def __len__(self):
        return self.length

def FromHDF5(data, key, inputs=1, batch_size=1, shuffle=True, sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=False):
    _dataset_loader = DataLoader(_HDF5Dataset(data, key),
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=drop_last)
    return _dataset_loader, inputs
