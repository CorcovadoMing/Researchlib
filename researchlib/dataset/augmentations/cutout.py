import numpy as np
from collections import namedtuple


# Interface
class Cutout(namedtuple('Cutout', ('h', 'w', 'cut'))):
    def __call__(self, x, choice, x0, y0):
        if choice:
            x[y0:y0 + self.cut, x0:x0 + self.cut, :] = 0
        return x

    def options(self, prob=0.5):
        W, H = self.w, self.h
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'x0': np.random.choice(range(W + 1 - self.cut)),
            'y0': np.random.choice(range(W + 1 - self.cut))
        }
