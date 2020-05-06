import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


# Interface
class Sparkle(namedtuple('Sparkle', ())):
    def __call__(self, x, y, choice, p, s):
        if choice:
            aug = iaa.CoarseDropout(p, size_percent=s)
            x = aug.augment_image(x)
        return x, y

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'p': np.random.random() * 0.5,
            's': np.random.random() * 0.5
        }

