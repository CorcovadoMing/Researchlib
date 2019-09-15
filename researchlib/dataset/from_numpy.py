import torch
import torch.utils.data as utils


def FromNumpy(data,
              inputs=1,
              batch_size=1,
              shuffle=True,
              sampler=None,
              batch_sampler=None,
              drop_last=False):
    data = [torch.from_numpy(i).float() for i in data]
    _dataset = utils.TensorDataset(*data)
    _shuffle = shuffle
    _sampler = None
    if sampler is not None:
        _shuffle = False
        _sampler = sampler(_dataset)
    _loader = utils.DataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=_shuffle,
        sampler=_sampler,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last)
    return _loader, inputs
