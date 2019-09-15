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
from .reshape import _Reshape
from .condition_projection import _ConditionProjection
from .spatial_transform import _SpatialTransform
from .permute import _Permute
from .shakedrop import _ShakeDrop
from .pixel_norm import _PixelNorm
from .drop_relu import _DropReLU
from .coord_conv import _CoordConvTranspose2d, _CoordConv2d
from .non_local.concatenation import _ConcatNonLocalBlock1d, _ConcatNonLocalBlock2d, _ConcatNonLocalBlock3d
from .non_local.dot_product import _DotNonLocalBlock1d, _DotNonLocalBlock2d, _DotNonLocalBlock3d
from .non_local.embedded_gaussian import _EmbeddedGaussianNonLocalBlock1d, _EmbeddedGaussianNonLocalBlock2d, _EmbeddedGaussianNonLocalBlock3d
from .non_local.gaussian import _GaussianNonLocalBlock1d, _GaussianNonLocalBlock2d, _GaussianNonLocalBlock3d
from .noise_injection import _NoiseInjection
from .flatten import _Flatten
from .mish import _Mish
from .multiply import _Multiply
from .aaconv import _AAConv2d
from .blur import Downsample as _Downsample
from .wsconv import _WSConv1d, _WSConvTranspose1d, _WSConv2d, _WSConvTranspose2d, _WSConv3d, _WSConvTranspose3d
from .manifold_mixup import _Manifold_Mixup

#from .act import * (need more implementation)
#from .multihead_attention import * (Buggy)


class layer(object):
    # Non-Local
    ConcatNonLocalBlock1d = _ConcatNonLocalBlock1d
    ConcatNonLocalBlock2d = _ConcatNonLocalBlock2d
    ConcatNonLocalBlock3d = _ConcatNonLocalBlock3d
    DotNonLocalBlock1d = _DotNonLocalBlock1d
    DotNonLocalBlock2d = _DotNonLocalBlock2d
    DotNonLocalBlock3d = _DotNonLocalBlock3d
    EmbeddedGaussianNonLocalBlock1d = _EmbeddedGaussianNonLocalBlock1d
    EmbeddedGaussianNonLocalBlock2d = _EmbeddedGaussianNonLocalBlock2d
    EmbeddedGaussianNonLocalBlock3d = _EmbeddedGaussianNonLocalBlock3d
    GaussianNonLocalBlock1d = _GaussianNonLocalBlock1d
    GaussianNonLocalBlock2d = _GaussianNonLocalBlock2d
    GaussianNonLocalBlock3d = _GaussianNonLocalBlock3d

    CoordConvTranspose2d = _CoordConvTranspose2d
    CoordConv2d = _CoordConv2d

    WSConv1d = _WSConv1d
    WSConvTranspose1d = _WSConvTranspose1d
    WSConv2d = _WSConv2d
    WSConvTranspose2d = _WSConvTranspose2d
    WSConv3d = _WSConv3d
    WSConvTranspose3d = _WSConvTranspose3d

    Multiply = _Multiply

    AAConv2d = _AAConv2d
    Downsample = _Downsample

    # Recurrent
    LSTM = _LSTM
    QRNN = _QRNN

    # Activator
    Swish = _Swish
    GeLU = _GeLU
    Mish = _Mish

    # DropBlock
    DropBlock2d = _DropBlock2d
    DropBlock3d = _DropBlock3d

    # Capsule Network
    CapsuleMasked = _CapsuleMasked
    RoutingCapsules = _RoutingCapsules
    PrimaryCapsules = _PrimaryCapsules

    # Binary Network
    BinarizeLinear = _BinarizeLinear
    BinarizeConv2d = _BinarizeConv2d

    # Variants Convolution
    OctConv2d = _OctConv2d

    # ShakeDrop
    ShakeDrop = _ShakeDrop

    # Manifold Mixup
    Manifold_Mixup = _Manifold_Mixup

    # Others
    Norm = _Norm
    AdaptiveConcatPool1d = _AdaptiveConcatPool1d
    AdaptiveConcatPool2d = _AdaptiveConcatPool2d
    PositionEncoding = _PositionEncoding
    NoisyLinear = _NoisyLinear
    PixelNorm = _PixelNorm

    Reshape = _Reshape
    Permute = _Permute

    ConditionProjection = _ConditionProjection
    SpatialTransform = _SpatialTransform

    DropReLU = _DropReLU

    NoiseInjection = _NoiseInjection
    Flatten = _Flatten


# Merge nn and layer module if it didn't cause conflict
from torch import nn
for i, j in nn.__dict__.items():
    try:
        if 'torch.nn.modules' in str(j) and str(i)[0].isupper():
            try:
                getattr(layer, i)
            except:
                setattr(layer, i, j)
    except:
        pass
