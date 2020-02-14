from .equalize_lr import _EqualizeLr
from .time_distributed import _TimeDistributed

class Wrapper(object):
    EqualizeLr = _EqualizeLr
    TimeDistributed = _TimeDistributed 