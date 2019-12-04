from ..ops import op
from ..dataset import loader, Augmentations
from ..models import Node


def CIFAR10():
    graph = {}
    
    _cifar10 = [
        Node('cifar10', op.Source(loader.TorchDataset('cifar10', True, True), loader.TorchDataset('cifar10', False, False))),
        Node('normalize', op.Normalize('static', (125.31, 122.95, 113.87), (62.99, 62.09, 66.70)), 'cifar10'),
        Node('preloop', op.Preloop(), 'normalize'),
        Node('augmentation', op.Augmentation([Augmentations.Crop(32, 32, 4),
                                              Augmentations.HFlip()]), 'preloop'),
        Node('generator', op.Generator(), 'augmentation'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    
    for i in _cifar10:
        graph.update(i)
    
    return graph



class Loader(object):
    CIFAR10 = CIFAR10