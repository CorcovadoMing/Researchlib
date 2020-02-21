from .equalize_lr import _EqualizeLr
from .time_distributed import _TimeDistributed
from .random_distill import _RandomDistill


class Wrapper(object):
    EqualizeLr = _EqualizeLr
    TimeDistributed = _TimeDistributed
    RandomDistill = _RandomDistill