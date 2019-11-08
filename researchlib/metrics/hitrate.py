from collections import namedtuple
import numpy as np


class Hitrate(namedtuple('Hitrate', ['classes'])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        matrix = np.zeros((2, self.classes))
        for i in range(len(x)):
            if x[i] == y[i]:
                matrix[0, y[i]] += 1
            matrix[1, y[i]] += 1
        return matrix

