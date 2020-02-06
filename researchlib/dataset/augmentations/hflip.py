import numpy as np
from collections import namedtuple


class HFlip(namedtuple('HFlip', ())):
    def __call__(self, x, choice):
        return x[:, ::-1, :] if choice else x

    def options(self):
        return [{'choice': b} for b in [True, False]]
