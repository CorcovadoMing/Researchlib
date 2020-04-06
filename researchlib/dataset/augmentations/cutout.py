import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


# Interface
class Cutout(namedtuple('Cutout', ())):
    def __call__(self, x, choice):
        if choice:
            aug = iaa.Cutout(size=0.25, fill_mode='constant', cval=(0, 255), fill_per_channel=0.5)
            x = aug.augment_image(x)
        return x

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
        }
