from .conv import _Conv
from .depthwise_separable_conv import _DepthwiseSeparableConv


class unit(object):
    Conv = _Conv
    DepthwiseSeparableConv = _DepthwiseSeparableConv
