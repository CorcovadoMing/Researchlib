from torchvision import datasets, transforms
import torch


def MNIST(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.1307,), (0.3081,)))
    return datasets.MNIST('../data', train=train, download=True, transform=transforms.Compose(tf))


def CIFAR10(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    return datasets.CIFAR10('../data', train=train, download=True, transform=transforms.Compose(tf))


def FromDataset(dataset, batch_size, shuffle=True, sampler=None, normalize=True, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=False, custom_tf=[]):
    loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    sampler=sampler,
                                    batch_sampler=batch_sampler,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)
    return loader
