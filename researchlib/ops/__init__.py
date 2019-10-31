# Layers
from .lstm import _LSTM
from .qrnn import _QRNN
from .dropblock import _DropBlock2d, _DropBlock3d
from .capsule_layer import _CapsuleMasked, _RoutingCapsules, _PrimaryCapsules
from .binarized import _BinarizeLinear, _BinarizeConv2d
from .octconv import _OctConv2d
from .norm import _Norm
from .adaptive_concat_pool import _AdaptiveConcatPool1d, _AdaptiveConcatPool2d
from .position_encoding import _PositionEncoding
from .noisy_linear import _NoisyLinear
from .condition_projection import _ConditionProjection
from .spatial_transform import _SpatialTransform
from .shakedrop import _ShakeDrop
from .pixel_norm import _PixelNorm
from .drop_relu import _DropReLU
from .coord_conv import _CoordConvTranspose2d, _CoordConv2d
from .non_local.concatenation import _ConcatNonLocalBlock1d, _ConcatNonLocalBlock2d, _ConcatNonLocalBlock3d
from .non_local.dot_product import _DotNonLocalBlock1d, _DotNonLocalBlock2d, _DotNonLocalBlock3d
from .non_local.embedded_gaussian import _EmbeddedGaussianNonLocalBlock1d, _EmbeddedGaussianNonLocalBlock2d, _EmbeddedGaussianNonLocalBlock3d
from .non_local.gaussian import _GaussianNonLocalBlock1d, _GaussianNonLocalBlock2d, _GaussianNonLocalBlock3d
from .noise_injection import _NoiseInjection
from .multiply import _Multiply
from .aaconv import _AAConv2d
from .blur import Downsample as _Downsample
from .wsconv import _WSConv1d, _WSConvTranspose1d, _WSConv2d, _WSConvTranspose2d, _WSConv3d, _WSConvTranspose3d
from .manifold_mixup import _ManifoldMixup
from .shake_batchnorm import _ShakeBatchNorm1d, _ShakeBatchNorm2d, _ShakeBatchNorm3d
from .sasa import _SASA2d
#from .act import * (need more implementation)
#from .multihead_attention import * (Buggy)

from .activator import _GeLU, _Mish, _Swish
from .meta import _MultiApply, _SupportFeatureConcat
from .shaping import _Flatten, _Reshape, _Resize, _View, _Permute
from .sequence import Conv1dRNN, Conv1dLSTM, Conv1dPeepholeLSTM, Conv1dGRU
from .sequence import Conv2dRNN, Conv2dLSTM, Conv2dPeepholeLSTM, Conv2dGRU
from .sequence import Conv3dRNN, Conv3dLSTM, Conv3dPeepholeLSTM, Conv3dGRU
from .sequence import Conv1dRNNCell, Conv1dLSTMCell, Conv1dPeepholeLSTMCell, Conv1dGRUCell
from .sequence import Conv2dRNNCell, Conv2dLSTMCell, Conv2dPeepholeLSTMCell, Conv2dGRUCell
from .sequence import Conv3dRNNCell, Conv3dLSTMCell, Conv3dPeepholeLSTMCell, Conv3dGRUCell

from .active_noise import _ActiveNoise
from .rcl import _RConv2d


class op(object):
    # Meta Learning
    MultiApply= _MultiApply
    SupportFeatureConcat = _SupportFeatureConcat
    
    # Shaping
    View = _View
    Reshape = _Reshape
    Permute = _Permute
    Flatten = _Flatten
    Resize = _Resize
    
    # Sequence
    Conv1dRNN = Conv1dRNN
    Conv1dLSTM = Conv1dLSTM
    Conv1dPeepholeLSTM = Conv1dPeepholeLSTM
    Conv1dGRU = Conv1dGRU
    Conv2dRNN = Conv2dRNN 
    Conv2dLSTM = Conv2dLSTM
    Conv2dPeepholeLSTM = Conv2dPeepholeLSTM
    Conv2dGRU = Conv2dGRU
    Conv3dRNN = Conv3dRNN
    Conv3dLSTM = Conv3dLSTM
    Conv3dPeepholeLSTM = Conv3dPeepholeLSTM
    Conv3dGRU = Conv3dGRU
    Conv1dRNNCell = Conv1dRNNCell
    Conv1dLSTMCell = Conv1dLSTMCell
    Conv1dPeepholeLSTMCell = Conv1dPeepholeLSTMCell
    Conv1dGRUCell = Conv1dGRUCell
    Conv2dRNNCell = Conv2dRNNCell
    Conv2dLSTMCell = Conv2dLSTMCell 
    Conv2dPeepholeLSTMCell = Conv2dPeepholeLSTMCell
    Conv2dGRUCell = Conv2dGRUCell
    Conv3dRNNCell = Conv3dRNNCell
    Conv3dLSTMCell = Conv3dLSTMCell
    Conv3dPeepholeLSTMCell = Conv3dPeepholeLSTMCell 
    Conv3dGRUCell = Conv3dGRUCell
    
    ActiveNoise = _ActiveNoise
    
    SASA2d = _SASA2d
    RConv2d = _RConv2d

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
    ManifoldMixup = _ManifoldMixup

    # Others
    Norm = _Norm
    AdaptiveConcatPool1d = _AdaptiveConcatPool1d
    AdaptiveConcatPool2d = _AdaptiveConcatPool2d
    PositionEncoding = _PositionEncoding
    NoisyLinear = _NoisyLinear
    PixelNorm = _PixelNorm
    ShakeBatchNorm1d = _ShakeBatchNorm1d
    ShakeBatchNorm2d = _ShakeBatchNorm2d
    ShakeBatchNorm3d = _ShakeBatchNorm3d

    ConditionProjection = _ConditionProjection
    SpatialTransform = _SpatialTransform

    DropReLU = _DropReLU

    NoiseInjection = _NoiseInjection


# Merge nn and layer module if it didn't cause conflict
from torch import nn
for i, j in nn.__dict__.items():
    try:
        if 'torch.nn.modules' in str(j) and str(i)[0].isupper():
            try:
                getattr(op, i)
            except:
                setattr(op, i, j)
    except:
        pass
