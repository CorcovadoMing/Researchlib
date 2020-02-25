from ...ops import op
from ...dataset import loader, Augmentations, Preprocessing
from ...models import Node


def ImageWoof(normalize=True, resize=128):
    if normalize:
        magic_mean, magic_std = (123.675, 111.28, 103.53), (58.395, 57.12, 57.375)
    else:
        magic_mean, magic_std = 128, 128
    _source = [
        Node('source', op.Source(loader.LFS.Classification.ImageWoofFull(True, True, resize),
                                 loader.LFS.Classification.ImageWoofFull(False, False, resize))),
        Node('generator', op.Generator(
            Preprocessing.set_normalizer('static', magic_mean, magic_std),
            Preprocessing.Layout('HWC', 'CHW'),
            Augmentations.CircularCrop(resize, resize, resize//2), 
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(resize, resize, resize//2),
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


def ImageNette(normalize=True, resize=128):
    if normalize:
        magic_mean, magic_std = (123.675, 111.28, 103.53), (58.395, 57.12, 57.375)
    else:
        magic_mean, magic_std = 128, 128
    _source = [
        Node('source', op.Source(loader.LFS.Classification.ImageNetteFull(True, True, resize),
                                 loader.LFS.Classification.ImageNetteFull(False, False, resize))),
        Node('generator', op.Generator(
            Preprocessing.set_normalizer('static', magic_mean, magic_std),
            Preprocessing.Layout('HWC', 'CHW'),
            Augmentations.CircularCrop(resize, resize, resize//2), 
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(resize, resize, resize//2),
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