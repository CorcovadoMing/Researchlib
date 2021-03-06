import numpy as np
from collections import namedtuple
import imgaug.augmenters as iaa


# Interface
class Jigsaw(namedtuple('Jigsaw', ())):
    def __call__(self, x, y, choice, r, c):
        if choice:
            aug = iaa.Jigsaw(nb_rows=r, nb_cols=c, max_steps=(1, 5))
            res = iaa.Resize({'height': x.shape[0], 'width': x.shape[1]})
            x = res.augment_image(aug.augment_image(x))
        return x, y

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'r': np.random.randint(2, 6),
            'c': np.random.randint(2, 6),
        }


