import torch
import torch.utils.data as utils


def FromNumpy(data, inputs=1, batch_size=1, shuffle=True, sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=False):
    data = [torch.from_numpy(i) for i in data]
    dataset_ = utils.TensorDataset(*data)
    dataset_loader_ = utils.DataLoader(dataset_,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        sampler=sampler,
                                        batch_sampler=batch_sampler,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        drop_last=drop_last)
    return dataset_loader_, inputs
