from .conv import _conv
from .non_local import _non_local
from .res2 import _res2
from .randwire import _randwire


class unit(object):
    conv = _conv
    non_local = _non_local
    res2 = _res2
    randwire = _randwire
    
