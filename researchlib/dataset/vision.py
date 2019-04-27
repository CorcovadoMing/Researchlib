from torchvision import datasets, transforms

def MNIST(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.1307,), (0.3081,)))
    return datasets.MNIST('../data', train=train, download=True, transform=transforms.Compose(tf))
    
def CIFAR10(train=True, normalize=True, custom_tf=[]):
    tf = custom_tf + [transforms.ToTensor()]
    if normalize: tf.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    return datasets.CIFAR10('../data', train=train, download=True, transform=transforms.Compose(tf))

class vision(object):
    MNIST=MNIST
    CIFAR10=CIFAR10
