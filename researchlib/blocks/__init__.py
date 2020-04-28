from .unit import unit
from .dawnfast import _DAWNBlock
from .resblock_bottleneck import _ResBottleneckBlock
from .resnext_bottleneck import _ResnextBottleneckBlock
from .resblock import _ResBlock
from .wide_resblock import _WideResBlock
from .vggblock import _VGGBlock
from .whitening_block import _WhiteningBlock
from .revblock import _RevBlock
from .randwireblock import _RandWireBlock
from .skblock import _SKBlock
from .skblock_bottleneck import _SKBottleneckBlock
from .dynamic_routing_cell import _DynamicRoutingCell

class block(object):
    WhiteningBlock = _WhiteningBlock
    DAWNBlock = _DAWNBlock
    ResBottleneckBlock = _ResBottleneckBlock
    ResnextBottleneckBlock = _ResnextBottleneckBlock
    ResBlock = _ResBlock
    WideResBlock = _WideResBlock
    VGGBlock = _VGGBlock
    RevBlock = _RevBlock
    RandWireBlock = _RandWireBlock
    SKBlock = _SKBlock
    SKBottleneckBlock = _SKBottleneckBlock
    DynamicRoutingCell = _DynamicRoutingCell