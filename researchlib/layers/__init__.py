# Layers
from .lstm import _LSTM
from .qrnn import _QRNN
from .dropblock import _DropBlock2d, _DropBlock3d
from .capsule_layer import _CapsuleMasked, _RoutingCapsules, _PrimaryCapsules
from .binarized import _BinarizeLinear, _BinarizeConv2d
from .octconv import _OctConv2d

class layer(object):
    LSTM=_LSTM
    QRNN=_QRNN
    
    # DropBlock
    DropBlock2d=_DropBlock2d
    DropBlock3d=_DropBlock3d
    
    # Capsule Network
    CapsuleMasked=_CapsuleMasked
    RoutingCapsules=_RoutingCapsules
    PrimaryCapsules=_PrimaryCapsules
    
    # Binary Network
    BinarizeLinear=_BinarizeLinear 
    BinarizeConv2d=_BinarizeConv2d
    
    # Variants Convolution
    OctConv2d=_OctConv2d



# Blocks
from .block import block

# Buggy
from .act import *
from .norm import *
from .adaptive_concat_pool import *
from .activator import *
from .wrapper import *
from .noisy_linear import *
from .multihead_attention import *
from .position_encoding import *