from ..ops import op
from ..dataset import loader, Augmentations
from ..models import Node


def CIFAR10(normalize=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (125.31, 122.95, 113.87), (62.99, 62.09, 66.70)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128, 128, 128), (128, 128, 128)), 'source')
    _source = [
        Node('source', op.Source(loader.TorchDataset('cifar10', True, True), loader.TorchDataset('cifar10', False, False))),
        _normalize,
        Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.CircularCrop(32, 32, 4), 
                                              Augmentations.HFlip()]), 'preloop'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)   
    return graph


def CIFAR100(normalize=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (129.3105, 124.0575, 112.4295), (68.1615, 65.382, 70.431)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128, 128, 128), (128, 128, 128)), 'source')
    _source = [
        Node('source', op.Source(loader.TorchDataset('cifar100', True, True), loader.TorchDataset('cifar100', False, False))),
        _normalize,
        Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.CircularCrop(32, 32, 4), 
                                              Augmentations.HFlip()]), 'preloop'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)   
    return graph


def MNIST(normalize=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (33.3285,), (78.5655,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.TorchDataset('mnist', True, True), loader.TorchDataset('mnist', False, False))),
        _normalize,
        Node('preloop', op.Preloop(), 'normalize'),
        Node('generator', op.Generator(), 'preloop'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def ImageWoofFull(normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (123.675, 111.28, 103.53), (58.395, 57.12, 57.375)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128, 128, 128), (128, 128, 128)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Classification.ImageWoofFull(True, True, resize),
                                 loader.LFS.Classification.ImageWoofFull(False, False, resize))),
        _normalize,
        #Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.CircularCrop(resize, resize, resize//8), 
                                              Augmentations.NonlinearJitter(),
                                              Augmentations.HFlip(),
                                              Augmentations.Cutout(resize, resize, resize//8)]), 'normalize'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def ImageNetteFull(normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (123.675, 111.28, 103.53), (58.395, 57.12, 57.375)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128, 128, 128), (128, 128, 128)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Classification.ImageNetteFull(True, True, resize), 
                                 loader.LFS.Classification.ImageNetteFull(False, False, resize))),
        _normalize,
        #Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.CircularCrop(resize, resize, resize//8),
                                              Augmentations.HFlip()]), 'normalize'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


class Dataset(object):
    CIFAR10 = CIFAR10
    MNIST = MNIST
    ImageWoofFull = ImageWoofFull
    ImageNetteFull = ImageNetteFull