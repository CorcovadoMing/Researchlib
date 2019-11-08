from collections import namedtuple


class Acc(namedtuple('Acc', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach()
        return x.eq(y).sum().float() / x.size(0)
