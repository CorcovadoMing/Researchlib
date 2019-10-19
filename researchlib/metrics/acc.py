from collections import namedtuple


class Acc(namedtuple('Acc', [])):
    def __call__(self, x, y):
        return x.eq(y).sum().float() / x.size(0)
