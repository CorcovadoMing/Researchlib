# Layers
from .lstm import _LSTM
from .qrnn import _QRNN
from .dropblock import _DropBlock2d, _DropBlock3d
from .capsule_layer import _CapsuleMasked, _RoutingCapsules, _PrimaryCapsules
from .binarized import _BinarizeLinear, _BinarizeConv2d
from .octconv import _OctConv2d
from .swish import _Swish
from .gelu import _GeLU
from .norm import _Norm
from .adaptive_concat_pool import _AdaptiveConcatPool1d, _AdaptiveConcatPool2d
from .position_encoding import _PositionEncoding
from .noisy_linear import _NoisyLinear
from .flatten import _Flatten
from .reshape import _Reshape
from .condition_projection import _ConditionProjection
from .spatial_transform import _SpatialTransform

#from .act import * (need more implementation)
#from .multihead_attention import * (Buggy)

class layer(object):
    # Recurrent
    LSTM=_LSTM
    QRNN=_QRNN
    
    # Activator
    Swish=_Swish
    GeLU=_GeLU
    
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
    
    # Others
    Norm=_Norm
    AdaptiveConcatPool1d=_AdaptiveConcatPool1d
    AdaptiveConcatPool2d=_AdaptiveConcatPool2d
    PositionEncoding=_PositionEncoding
    NoisyLinear=_NoisyLinear

    Flatten=_Flatten
    Reshape=_Reshape
    
    ConditionProjection=_ConditionProjection
    SpatialTransform=_SpatialTransform

# Blocks
from .block import block

# Wrapper
from .wrapper import *