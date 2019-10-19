from .conv import _conv
from .res2 import _res2
from .randwire import _randwire
from .norm_act import _norm_act


class unit(object):
    conv = _conv
    res2 = _res2
    randwire = _randwire
    norm_act = _norm_act
