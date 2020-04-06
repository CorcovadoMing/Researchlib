import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


class Rotate(namedtuple('Rotate', ())):
    def __call__(self, x, choice):
        if choice:
            aug = iaa.pillike.Affine(rotate=(-45, 45), fillcolor=(0, 256))
            x = aug.augment_image(x.astype(np.uint8)).astype(np.float32)
        return x

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
        }

