import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


# Interface
class Cutout(namedtuple('Cutout', ('fillcolor'))):
    def __call__(self, x, choice):
        if choice:
            if self.fillcolor:
                aug = iaa.Cutout(size=0.5, fill_mode='constant', cval=(0, 255), fill_per_channel=0.5)
            else:
                aug = iaa.Cutout(size=0.5, fill_mode='constant', cval=0)
            x = aug.augment_image(x)
        return x

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
        }
