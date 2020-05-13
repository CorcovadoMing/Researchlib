from ...dataset import Augmentations

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Rotate, PadIfNeeded,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, RandomContrast, RandomBrightness, RandomCrop,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomGamma, ChannelShuffle, CoarseDropout, Blur, RGBShift,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, InvertImg, ToGray, GridDropout, NoOp, Solarize, Posterize, IAAPerspective, Equalize
)


def _FastWeakAugment2d(img_size, set='general'):
    _sets = {
        'general': [
            HorizontalFlip(p=0.5),
            Compose([
                PadIfNeeded(min_height=img_size*2, min_width=img_size*2, border_mode=cv2.BORDER_WRAP, p=1),
                RandomCrop(img_size, img_size, p=1),
            ], p=0.5),
        ],
    }
    
    return _sets[set]
