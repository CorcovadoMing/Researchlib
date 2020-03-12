from ...dataset import Augmentations


def _WeakAugment2d(img_size):
    return [
        Augmentations.CircularCrop(img_size, img_size, img_size//2), 
        Augmentations.HFlip(),
    ]
