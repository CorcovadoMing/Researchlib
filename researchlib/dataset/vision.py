from torchvision import datasets, transforms

def _MNIST(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.5,), (0.5,)))
    return datasets.MNIST('../data', train=train, download=True, transform=transforms.Compose(tf))

def _KMNIST(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.5,), (0.5,)))
    return datasets.KMNIST('../data', train=train, download=True, transform=transforms.Compose(tf))

def _FashionMNIST(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.5,), (0.5,)))
    return datasets.FashionMNIST('../data', train=train, download=True, transform=transforms.Compose(tf))

def _CIFAR10(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return datasets.CIFAR10('../data', train=train, download=True, transform=transforms.Compose(tf))