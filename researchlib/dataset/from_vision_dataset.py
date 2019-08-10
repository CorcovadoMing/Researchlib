import torch
from .fast_collate import fast_collate

def VisionDataset(dataset,
                  batch_size,
                  train=True,
                  sampler=None,
                  batch_sampler=None,
                  num_workers=2,
                  pin_memory=True,
                  drop_last=False):
    _dataset = dataset(train)
    _shuffle = train
    _sampler = None
    if sampler is not None:
        _shuffle = False
        _sampler = sampler(_dataset)
    loader = torch.utils.data.DataLoader(_dataset,
                                         batch_size=batch_size,
                                         shuffle=_shuffle,
                                         sampler=_sampler,
                                         batch_sampler=batch_sampler,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory,
                                         drop_last=drop_last,
                                         collate_fn=fast_collate)
    return loader
