from ..ops import op
from ..dataset import loader, Augmentations, Preprocessing
from ..models import Node


def Dice(name, normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.Dice(name, True, resize), 
                                 loader.LFS.Restoration.Dice('1000nactest2', False, resize))),
        _normalize,
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def Noise2d(name, normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.Noise2d(name, True, True, resize), 
                                 loader.LFS.Restoration.Noise2d(name, True, False, resize))),
        _normalize,
        #Node('preloop', op.Preloop(), 'normalize'),
        #Node('augmentation', op.Augmentation([Augmentations.CircularCrop(resize, resize, resize//4),
        #                                      Augmentations.HFlip()]), 'normalize'),
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def BreastCancer(normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Medical.BreastCancer(True, resize), 
                                 loader.LFS.Medical.BreastCancer(True, resize))),
        _normalize,
        #Node('preloop', op.Preloop(), 'normalize'),
        #Node('augmentation', op.Augmentation([Augmentations.CircularCrop(resize, resize, resize//4),
        #                                      Augmentations.HFlip()]), 'normalize'),
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def RENOIR(normalize=True, resize=128):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,128,128), (128,128,128)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,128,128), (128,128,128)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.RENOIR(True, resize), 
                                 loader.LFS.Restoration.RENOIR(True, resize))),
        _normalize,
        #Node('preloop', op.Preloop(), 'normalize'),
        #Node('augmentation', op.Augmentation([Augmentations.CircularCrop(resize, resize, resize//4),
        #                                      Augmentations.HFlip()]), 'normalize'),
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def Test(name, normalize=True, resize=256):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
        
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.Test(name, True, True, resize),
                                 loader.LFS.Restoration.Test(name, False, False, resize))),
        _normalize,
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def COIL100(name, normalize=True, resize=256, merge_train_val=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.COIL100(name, True, True, merge_train_val, resize),
                                 loader.LFS.Restoration.COIL100(name, False, False, merge_train_val, resize))),
        _normalize,
        Node('generator', op.Generator(), 'normalize'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    for i in _source:
        node, node_type = i
        graph.update(node)
    return graph


def BSD68(name, normalize=True, resize=None, merge_train_val=True):
    graph = {}
    if normalize:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    else:
        _normalize = Node('normalize', op.Normalize('static', (128,), (128,)), 'source')
    _source = [
        Node('source', op.Source(loader.LFS.Restoration.BSD68(name, True, True, merge_train_val, resize),
                                 loader.LFS.Restoration.BSD68(name, False, False, merge_train_val, resize))),
        _normalize,
        Node('generator', op.Generator(), 'normalize'),
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
    
    Dice = Dice
    Noise2d = Noise2d
    
    BreastCancer = BreastCancer
    RENOIR = RENOIR
    Test = Test
    COIL100 = COIL100
    BSD68 = BSD68