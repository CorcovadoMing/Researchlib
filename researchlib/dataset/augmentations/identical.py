from collections import namedtuple


class Identical(namedtuple('Identical', ())):
    def __call__(self, x, choice):
        return x

    def options(self):
        return [{'choice': b} for b in [True, False]]
