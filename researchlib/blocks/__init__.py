from .unit import unit
from .dawnfast import _DAWNBlock
from .resblock_bottleneck import _ResBottleneckBlock
from .resblock import _ResBlock
from .wide_resblock import _WideResBlock
from .vggblock import _VGGBlock
from .whitening_block import _WhiteningBlock
from .revblock import _RevBlock

class block(object):
    WhiteningBlock = _WhiteningBlock
    DAWNBlock = _DAWNBlock
    ResBottleneckBlock = _ResBottleneckBlock
    ResBlock = _ResBlock
    WideResBlock = _WideResBlock
    VGGBlock = _VGGBlock
    RevBlock = _RevBlock