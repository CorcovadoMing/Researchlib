from torchvision import datasets


def _MNIST(train=True):
    return datasets.MNIST('../data',
                          train=train,
                          download=True)


def _KMNIST(train=True):
    return datasets.KMNIST('../data',
                           train=train,
                           download=True)


def _FashionMNIST(train=True):
    return datasets.FashionMNIST('../data',
                                 train=train,
                                 download=True)


def _CIFAR10(train=True):
    return datasets.CIFAR10('../data',
                            train=train,
                            download=True)
