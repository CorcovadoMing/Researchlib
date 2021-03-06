from .dropblock import _DropBlock2d, _DropBlock3d
from .capsule_layer import _CapsuleMasked, _RoutingCapsules, _PrimaryCapsules
from .binarized import _BinarizeLinear, _BinarizeConv2d
from .octconv import _OctConv2d
from .norm import _Norm
from .adaptive_concat_pool import _AdaptiveConcatPool1d, _AdaptiveConcatPool2d
from .noisy_linear import _NoisyLinear
from .condition_projection import _ConditionProjection
from .spatial_transform import _SpatialTransform
from .shakedrop import _ShakeDrop
from .pixel_norm import _PixelNorm

from .multiply import _Multiply
from .aaconv import _AAConv2d
from .blur import Downsample as _Downsample
from .manifold_mixup import _ManifoldMixup
from .shake_batchnorm import _ShakeBatchNorm1d, _ShakeBatchNorm2d, _ShakeBatchNorm3d
#from .act import * (need more implementation)
#from .multihead_attention import * (Buggy)

from .template_bank import _TemplateBank

#============================================================================

from .conv_variants import _SConv2d, _SepConv2d, _DilConv2d, _DiracConv1d, _DiracConv2d, _DiracConv3d, _PacConv2d, _PacConvTranspose2d, _PacPool2d, _SKConv2d, _RConv2d, _SASA2d, _GhostConv2d, _FastDeconv2d

from .activator import _GeLU, _Mish, _Swish, _DropReLU, _LogSoftmax, _Softmax, _GumbelSoftmax, _LogSparsemax, _Sparsemax, _FTSwishPlus, _TReLU

from .ghost_batchnorm import _GhostBatchNorm2d

from .meta import _MultiApply, _SupportFeatureConcat

from .shaping import _Flatten, _FlattenExcept, _Reshape, _Resize, _View, _Permute, _OneHot

from .sequence import Conv1dRNN, Conv1dLSTM, Conv1dPeepholeLSTM, Conv1dGRU
from .sequence import Conv2dRNN, Conv2dLSTM, Conv2dPeepholeLSTM, Conv2dGRU
from .sequence import Conv3dRNN, Conv3dLSTM, Conv3dPeepholeLSTM, Conv3dGRU
from .sequence import Conv1dRNNCell, Conv1dLSTMCell, Conv1dPeepholeLSTMCell, Conv1dGRUCell
from .sequence import Conv2dRNNCell, Conv2dLSTMCell, Conv2dPeepholeLSTMCell, Conv2dGRUCell
from .sequence import Conv3dRNNCell, Conv3dLSTMCell, Conv3dPeepholeLSTMCell, Conv3dGRUCell
from .sequence import _QRNN, _LSTM, _Seq2Seq, _Set2Set, _ALSTM

from .encoding import _PositionalEncoding1d, _CoordinatesEncoding2d

from .non_local import _ConcatNonLocalBlock1d, _ConcatNonLocalBlock2d, _ConcatNonLocalBlock3d
from .non_local import _DotNonLocalBlock1d, _DotNonLocalBlock2d, _DotNonLocalBlock3d
from .non_local import _EmbeddedGaussianNonLocalBlock1d, _EmbeddedGaussianNonLocalBlock2d, _EmbeddedGaussianNonLocalBlock3d
from .non_local import _GaussianNonLocalBlock1d, _GaussianNonLocalBlock2d, _GaussianNonLocalBlock3d

from .active_noise import _ActiveNoise
from .inputs import _SetVariable, _UpdateVariable, _GetVariable, _Source, _Generator, _Preloop, _GeneratorDev
from .node import _To, _Subgraph, _Detach, _Name, _NoOp, _Identical

from .nonparams import _Flip, _Average, _WeightedAverage, _Add, _Sum, _Rotation42d, _Anneal, _Argmax, _Argmin, _Mixture, _RPT, _SobelHorizontal2d, _SobelVertical2d, _Perception, _Gaussian2d
from .pool import _CombinePool2d, _PixelShuffle2d

from .mlp import _MLP

from .n2v import N2V
from .wrapper import Wrapper
from .prob import Prob
from .rl import RL
from .buffer import Buffer
from .checker import Checker

from .prepare import _PrepareImage2d, _RandAugment2d, _WeakAugment2d, _FastRandAugment2d, _FastWeakAugment2d

from .pyramidnet import _PyramidNet110, _PyramidNet272
from .vgg import _VGG19
from .resnet import _ResNet, _ResNet18, _ResNet50
from .preresnet import _PreResNet50
from .wideresnet import _WideResNet28x10
from .decoder import _AEDecoder2d, _VAEDecoder2d
from .coco import _COCO

from .presknet import _PreSKNet50
from .sknet import _SKNet50, _SKNet18
from .frn import _FRN2d
from .tlu import _TLU2d

from .fuse_sampling import _FuseSampling2d

from .storage import Storage


class op(object):
    Storage = Storage
    
    GeneratorDev = _GeneratorDev
    FastRandAugment2d = _FastRandAugment2d
    FastWeakAugment2d = _FastWeakAugment2d
    
    FRN2d = _FRN2d
    TLU2d = _TLU2d
    FuseSampling2d = _FuseSampling2d
    
    FastDeconv2d = _FastDeconv2d
    
    # Submodules
    N2V = N2V
    Wrapper = Wrapper
    Prob = Prob
    RL = RL
    Buffer = Buffer
    Checker = Checker
    
    MLP = _MLP
    ResNet = _ResNet
    ResNet18 = _ResNet18
    ResNet50 = _ResNet50
    PreResNet50 = _PreResNet50
    WideResNet28x10 = _WideResNet28x10
    PreSKNet50 = _PreSKNet50
    SKNet50 = _SKNet50
    SKNet18 = _SKNet18
    VGG19 = _VGG19
    PyramidNet110 = _PyramidNet110
    PyramidNet272 = _PyramidNet272
    
    AEDecoder2d = _AEDecoder2d
    VAEDecoder2d = _VAEDecoder2d
    
    TemplateBank = _TemplateBank
    SConv2d = _SConv2d
    SepConv2d = _SepConv2d
    DilConv2d = _DilConv2d
    DiracConv1d = _DiracConv1d
    DiracConv2d = _DiracConv2d
    DiracConv3d = _DiracConv3d
    
    COCO = _COCO
    
    # Prepare
    PrepareImage2d = _PrepareImage2d 
    RandAugment2d = _RandAugment2d
    WeakAugment2d = _WeakAugment2d
    
    # Pool
    CombinePool2d = _CombinePool2d
    PixelShuffle2d = _PixelShuffle2d
    
    # Non-Params
    Flip = _Flip
    Average = _Average
    WeightedAverage = _WeightedAverage
    Sum = _Sum
    Add = _Add
    Rotation42d = _Rotation42d
    Anneal = _Anneal
    Argmax = _Argmax
    Argmin = _Argmin
    Mixture = _Mixture
    RPT = _RPT
    SobelHorizontal2d = _SobelHorizontal2d
    SobelVertical2d = _SobelVertical2d
    Perception = _Perception
    Gaussian2d = _Gaussian2d
    
    # inputs
    SetVariable = _SetVariable
    UpdateVariable = _UpdateVariable
    GetVariable = _GetVariable
    Source = _Source
    Generator = _Generator
    Preloop = _Preloop
    
    # Encoding
    PositionalEncoding1d = _PositionalEncoding1d
    CoordinatesEncoding2d = _CoordinatesEncoding2d
    
    # Meta Learning
    MultiApply= _MultiApply
    SupportFeatureConcat = _SupportFeatureConcat
    
    # Node
    To = _To
    Subgraph = _Subgraph
    Detach = _Detach
    Name = _Name
    NoOp = _NoOp
    Identical = _Identical
    
    # Shaping
    View = _View
    Reshape = _Reshape
    Permute = _Permute
    Flatten = _Flatten
    FlattenExcept = _FlattenExcept
    Resize = _Resize
    OneHot = _OneHot
    
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
    
    LSTM = _LSTM
    QRNN = _QRNN
    ALSTM = _ALSTM
    
    Seq2Seq = _Seq2Seq
    Set2Set = _Set2Set
    
    ActiveNoise = _ActiveNoise
    

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

    Multiply = _Multiply

    AAConv2d = _AAConv2d
    Downsample = _Downsample

    # Activator
    Swish = _Swish
    GeLU = _GeLU
    Mish = _Mish
    DropReLU = _DropReLU
    LogSoftmax = _LogSoftmax
    Softmax = _Softmax
    GumbelSoftmax = _GumbelSoftmax
    LogSparsemax = _LogSparsemax
    Sparsemax = _Sparsemax
    FTSwishPlus = _FTSwishPlus
    TReLU = _TReLU

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
    PacConv2d = _PacConv2d
    PacConvTranspose2d = _PacConvTranspose2d
    PacPool2d = _PacPool2d
    SKConv2d = _SKConv2d
    SASA2d = _SASA2d
    RConv2d = _RConv2d
    GhostConv2d = _GhostConv2d

    # ShakeDrop
    ShakeDrop = _ShakeDrop

    # Manifold Mixup
    ManifoldMixup = _ManifoldMixup

    # Others
    Norm = _Norm
    AdaptiveConcatPool1d = _AdaptiveConcatPool1d
    AdaptiveConcatPool2d = _AdaptiveConcatPool2d
    NoisyLinear = _NoisyLinear
    PixelNorm = _PixelNorm
    ShakeBatchNorm1d = _ShakeBatchNorm1d
    ShakeBatchNorm2d = _ShakeBatchNorm2d
    ShakeBatchNorm3d = _ShakeBatchNorm3d
    ConditionProjection = _ConditionProjection
    SpatialTransform = _SpatialTransform
    GhostBatchNorm2d = _GhostBatchNorm2d



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
