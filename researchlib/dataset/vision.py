from torchvision import datasets, transforms


def _MNIST(train = True, custom_tf = []):
    tf = custom_tf + [transforms.ToTensor()]
    return datasets.MNIST(
        '../data', train = train, download = True, transform = transforms.Compose(tf)
    )


def _KMNIST(train = True, custom_tf = []):
    tf = custom_tf + [transforms.ToTensor()]
    return datasets.KMNIST(
        '../data', train = train, download = True, transform = transforms.Compose(tf)
    )


def _FashionMNIST(train = True, custom_tf = []):
    tf = custom_tf + [transforms.ToTensor()]
    return datasets.FashionMNIST(
        '../data', train = train, download = True, transform = transforms.Compose(tf)
    )


def _CIFAR10(train = True, custom_tf = []):
    tf = custom_tf + [transforms.ToTensor()]
    return datasets.CIFAR10(
        '../data', train = train, download = True, transform = transforms.Compose(tf)
    )


def _CIFAR100(train = True, custom_tf = []):
    tf = custom_tf + [transforms.ToTensor()]
    return datasets.CIFAR100(
        '../data', train = train, download = True, transform = transforms.Compose(tf)
    )
