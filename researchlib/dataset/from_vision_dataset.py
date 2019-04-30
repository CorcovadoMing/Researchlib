import torch

def VisionDataset(dataset, batch_size, train=True, sampler=None, normalize=True, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=False, custom_tf=[]):
    loader = torch.utils.data.DataLoader(dataset(train, normalize=normalize, custom_tf=custom_tf),
                                    batch_size=batch_size,
                                    shuffle=train,
                                    sampler=sampler,
                                    batch_sampler=batch_sampler,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)
    return loader
