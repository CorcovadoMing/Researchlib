from ...ops import op
from ...dataset import loader, Augmentations, Preprocessing
from ...models import Node


def CIFAR10(normalize=True):
    if normalize:
        magic_mean, magic_std = (125.31, 122.95, 113.87), (62.99, 62.09, 66.70)
    else:
        magic_mean, magic_std = 128, 128
    _source = [
        Node('source', op.Source(loader.TorchDataset('cifar10', True, True), 
                                 loader.TorchDataset('cifar10', False, False))),
        Node('generator', op.Generator(
            Preprocessing.set_normalizer('static', magic_mean, magic_std),
            Preprocessing.Layout('HWC', 'CHW'),
            Augmentations.CircularCrop(32, 32, 16), 
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(32, 32, 16),
            Augmentations.Invert(),
            Augmentations.Equalize(),
            Augmentations.Identical(),
            Augmentations.ShearX(),
            Augmentations.ShearY(),
            Augmentations.TranslateX(),
            Augmentations.TranslateY(),
            Augmentations.Rotate(),
        #     Augmentations.Solarize(),
            Augmentations.Posterize(),
            Augmentations.Contrast(),
            Augmentations.Color(),
            Augmentations.Brightness(),
            Augmentations.Sharpness()
        ), 'source'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    return _source




def CIFAR100(normalize=True):
    if normalize:
        magic_mean, magic_std = (129.3105, 124.0575, 112.4295), (68.1615, 65.382, 70.431)
    else:
        magic_mean, magic_std = 128, 128
    _source = [
        Node('source', op.Source(loader.TorchDataset('cifar100', True, True), 
                                 loader.TorchDataset('cifar100', False, False))),
        Node('generator', op.Generator(
            Preprocessing.set_normalizer('static', magic_mean, magic_std),
            Preprocessing.Layout('HWC', 'CHW'),
            Augmentations.CircularCrop(32, 32, 16), 
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(32, 32, 16),
            Augmentations.Invert(),
            Augmentations.Equalize(),
            Augmentations.Identical(),
            Augmentations.ShearX(),
            Augmentations.ShearY(),
            Augmentations.TranslateX(),
            Augmentations.TranslateY(),
            Augmentations.Rotate(),
        #     Augmentations.Solarize(),
            Augmentations.Posterize(),
            Augmentations.Contrast(),
            Augmentations.Color(),
            Augmentations.Brightness(),
            Augmentations.Sharpness()
        ), 'source'),
        Node('x', op.Name(), 'generator:0'),
        Node('y', op.Name(), 'generator:1'),
    ]
    return _source



