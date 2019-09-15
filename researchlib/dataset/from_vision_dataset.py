import torch


def VisionDataset(dataset,
                  batch_size,
                  train=True,
                  sampler=None,
                  batch_sampler=None,
                  drop_last=False,
                  custom_tf=[]):
    _dataset = dataset(train, custom_tf=custom_tf)
    _shuffle = train
    _sampler = None
    if sampler is not None:
        _shuffle = False
        _sampler = sampler(_dataset)
    loader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=_shuffle,
        sampler=_sampler,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last)
    return loader
