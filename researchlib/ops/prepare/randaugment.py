from ...dataset import Augmentations


def _RandomAugment2d(img_size):
    return [
        Augmentations.CircularCrop(img_size, img_size, img_size//2),
        Augmentations.HFlip(),
        Augmentations.AutoContrast(),
        Augmentations.Cutout(),
        Augmentations.Invert(),
        Augmentations.Equalize(),
        Augmentations.Identical(),
        Augmentations.ShearX(),
        Augmentations.ShearY(),
        Augmentations.TranslateX(),
        Augmentations.TranslateY(),
        Augmentations.Rotate(),
        Augmentations.Solarize(),
        Augmentations.Posterize(),
        Augmentations.Contrast(),
        Augmentations.Color(),
        Augmentations.Brightness(),
        Augmentations.Sharpness(),
        Augmentations.SVDBlur(img_size),
        Augmentations.Sparkle(),
        Augmentations.Scale(),
        Augmentations.Jigsaw(),
    ]