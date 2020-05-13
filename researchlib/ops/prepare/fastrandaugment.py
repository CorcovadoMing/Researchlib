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

import numpy as np

_color_fill = [
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (128, 128, 128),
]


def _FastRandAugment2d(img_size, set='general'):
    _sets = {
        'general': [
            OneOf([
                Blur(p=1),
                GaussNoise(p=1),
                MotionBlur(p=1),
                MedianBlur(p=1),
            ]),
            HorizontalFlip(p=1),
            Compose([
                PadIfNeeded(min_height=img_size*2, min_width=img_size*2, border_mode=cv2.BORDER_WRAP, p=1),
                RandomCrop(img_size, img_size, p=1),
            ], p=1),
            RandomBrightnessContrast(p=1),
            InvertImg(p=1),
            ToGray(p=1),
            RGBShift(p=1),
            OneOf([
                GridDropout(holes_number_x=4, holes_number_y=4, random_offset=True, ratio=r, fill_value=i, p=1) for i in _color_fill for r in np.linspace(0.3, 0.7, 4)
            ], p=1),
            NoOp(p=1),
            Equalize(p=1),
            Rotate(45, border_mode=cv2.BORDER_WRAP, p=1),
            ShiftScaleRotate(shift_limit=0.25, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_WRAP, p=1),
            Solarize(p=1),
            Posterize(p=1),
            IAAPerspective(p=1),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
            RandomGamma(p=1),   
            ChannelShuffle(p=1),
            OneOf([
                CoarseDropout(max_holes=1, max_height=img_size//2, max_width=img_size//2, fill_value=i, p=1) for i in _color_fill
            ], p=1),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
            ], p=1),
        ],
    }
    
    return _sets[set]
