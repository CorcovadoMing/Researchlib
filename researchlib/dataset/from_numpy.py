import torch
import torch.utils.data as utils


def FromNumpy(x, y, batch_size=1, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False):
    x_ = torch.from_numpy(x).float()
    y_ = torch.from_numpy(y).float()
    dataset_ = utils.TensorDataset(x_, y_)
    dataset_loader_ = utils.DataLoader(dataset_,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        sampler=sampler,
                                        batch_sampler=batch_sampler,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        drop_last=drop_last)
    return dataset_loader_
