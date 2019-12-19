from ..ops import op
from ..dataset import loader, Augmentations
from ..models import Node


def CIFAR10(normalize=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (125.31, 122.95, 113.87), (62.99, 62.09, 66.70)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128, 128, 128), (128, 128, 128)), 'source')
    _cifar10 = [
        Node('source', op.Source(loader.TorchDataset('cifar10', True, True), loader.TorchDataset('cifar10', False, False))),
        _normalize,
        Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.Crop(32, 32, 4), Augmentations.HFlip()]), 'preloop'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _cifar10:
        node, node_type = i
        graph.update(node)   
    return graph


def MNIST(normalize=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (33.3285,), (78.5655,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _mnist = [
        Node('source', op.Source(loader.TorchDataset('mnist', True, True), loader.TorchDataset('mnist', False, False))),
        _normalize,
        Node('preloop', op.Preloop(), 'normalize'),
        Node('generator', op.Generator(), 'preloop'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _mnist:
        node, node_type = i
        graph.update(node)
    return graph


class Dataset(object):
    CIFAR10 = CIFAR10
    MNIST = MNIST