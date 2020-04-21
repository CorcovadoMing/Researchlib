from ...ops import op
from ...dataset import loader, Augmentations, Preprocessing
from ...models import Node


def MNIST(normalize=True):
    if normalize:
        magic_mean, magic_std = (33.3285,), (78.5655,)
    else:
        magic_mean, magic_std = 128, 128
    _source = [
        Node('source', op.Source(loader.TorchDataset('mnist', True, True), 
                                 loader.TorchDataset('mnist', False, False))),
        Node('generator', op.Generator(
            Preprocessing.set_normalizer('static', magic_mean, magic_std),
            Preprocessing.Layout('HWC', 'CHW'),
        ), 'source'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    return _source