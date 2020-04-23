from .conv import _Conv
from .conv_frn import _ConvFRN

class unit(object):
    Conv = _Conv
    ConvFRN = _ConvFRN
