from torchvision import datasets, transforms
import torch


def FromPublic(dataset_name, mode='train', batch_size=1, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False):
    train_ = True
    if mode != 'train':
        train_ = False

    if dataset_name == 'mnist':
        dataset = datasets.MNIST('../data', train=train_, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10('../data', train=train_, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
                           
    loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    sampler=sampler,
                                    batch_sampler=batch_sampler,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)
    return loader