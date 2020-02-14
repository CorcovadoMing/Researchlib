import numpy as np
from collections import namedtuple


class Identical(namedtuple('Identical', ())):
    def __call__(self, x, choice):
        return x

    def options(self, prob=0.5):
        return [{'choice': b} for b in np.random.choice([True, False], p=[prob, 1-prob], size=1)]
