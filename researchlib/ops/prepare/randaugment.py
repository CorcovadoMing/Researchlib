from ...dataset import Augmentations


def _RandAugment2d(img_size, set='general'):
    _sets = {
        'general': [
            Augmentations.CircularCrop(img_size, img_size, img_size//2),
            Augmentations.HFlip(),
            Augmentations.AutoContrast(),
            Augmentations.Cutout(fillcolor=True),
            Augmentations.Invert(),
            Augmentations.Equalize(),
            Augmentations.Identical(),
            Augmentations.ShearX(),
            Augmentations.ShearY(),
            Augmentations.TranslateX(),
            Augmentations.TranslateY(),
            Augmentations.Rotate(fillcolor=True),
            Augmentations.Solarize(),
            Augmentations.Posterize(),
            Augmentations.Contrast(),
            Augmentations.Color(),
            Augmentations.Brightness(),
            Augmentations.Sharpness(),
            Augmentations.SVDBlur(img_size),
            Augmentations.Sparkle(),
            Augmentations.Scale(),
            Augmentations.Jigsaw()
        ],
    }
    
    return _sets[set]