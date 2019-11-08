from collections import namedtuple


class _Detach(namedtuple('Detach', [])):
    def __call__(self, x):
        return x.detach()

