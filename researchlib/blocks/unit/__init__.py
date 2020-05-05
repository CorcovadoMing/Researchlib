from .conv import _Conv
from .deconv import _Deconv

class unit(object):
    Conv = _Conv
    Deconv = _Deconv
