import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


# Interface
class Scale(namedtuple('Scale', ())):
    def __call__(self, x, y, choice, s):
        if choice:
            aug = iaa.Affine(scale=(1-s, 1+s))
            x = aug.augment_image(x)
        return x, y

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            's': 0.5
        }

