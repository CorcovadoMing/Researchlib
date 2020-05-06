import numpy as np
from collections import namedtuple


class CircularCrop(namedtuple('CircularCrop', ('h', 'w', 'pad'))):
    def __call__(self, x, y, choice, x0, y0):
        if choice:
            x = np.pad(x, ((self.pad, self.pad), (self.pad, self.pad), (0,0)), 'wrap')
            x = x[y0:y0 + self.h, x0:x0 + self.w, :]
        return x, y

    def options(self, prob=0.5):
        W, H = self.w + (2 * self.pad), self.h + (2 * self.pad)
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'x0': np.random.choice(range(W + 1 - self.w)),
            'y0': np.random.choice(range(H + 1 - self.h))
        }

