import numpy as np
from collections import namedtuple


class Identical(namedtuple('Identical', ())):
    def __call__(self, x, y, choice):
        return x, y

    def options(self, prob=0.5):
        return {'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1)}
