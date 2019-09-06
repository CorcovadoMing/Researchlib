# from .octconvblock import _OctConvBlock2d
# from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
# from .resblock import _ResBlock2d, _ResTransposeBlock2d
# from .resnextblock import _ResNextBlock2d, _ResNextTransposeBlock2d
# from .attentionblock import _AttentionBlock2d, _AttentionTransposeBlock2d
# from .denseblock import _DenseBlock2d, _DenseTransposeBlock2d
# from .basic_components import _CombinedDownSampling, _MaxPoolDownSampling, _InterpolateUpSampling, _ConvTransposeUpSampling


from .template.block import _Block as _TemplateBlock
from .dawnfast import _DAWNBlock
from .inception_residual_v2 import _InceptionResidualV2A, _ReductionV2A, _InceptionResidualV2B, _ReductionV2B, _InceptionResidualV2C
from .inception_v3 import _InceptionV3A, _InceptionV3B, _InceptionV3C, _InceptionV3D, _InceptionV3E
from .inception_v4 import _InceptionV4A, _ReductionV4A, _InceptionV4B, _ReductionV4B, _InceptionV4C
from .inverted_bottleneck import _InvertedBottleneckBlock
from .resblock_bottleneck import _ResBottleneckBlock
from .resblock import _ResBlock
from .vggblock import _VGGBlock
from .wide_resblock import _WideResBlock
from .unit import unit


class block(object):
    TemplateBlock = _TemplateBlock
    
    DAWNBlock = _DAWNBlock
    
    InceptionResidualV2A = _InceptionResidualV2A 
    ReductionV2A = _ReductionV2A
    InceptionResidualV2B = _InceptionResidualV2B
    ReductionV2B = _ReductionV2B
    InceptionResidualV2C = _InceptionResidualV2C
    
    InceptionV3A = _InceptionV3A
    InceptionV3B = _InceptionV3B
    InceptionV3C = _InceptionV3C
    InceptionV3D = _InceptionV3D
    InceptionV3E = _InceptionV3E
    
    InceptionV4A = _InceptionV4A
    ReductionV4A = _ReductionV4A
    InceptionV4B = _InceptionV4B
    ReductionV4B = _ReductionV4B
    InceptionV4C = _InceptionV4C
    
    InvertedBottleneckBlock = _InvertedBottleneckBlock
    
    ResBottleneckBlock = _ResBottleneckBlock
    
    ResBlock = _ResBlock
    
    VGGBlock = _VGGBlock
    
    WideResBlock = _WideResBlock
    
    
    
    

# class downsampling(object):
#     Combined = _CombinedDownSampling
#     MaxPool = _MaxPoolDownSampling


# class upsampling(object):
#     Interpolate = _InterpolateUpSampling
#     ConvTranspose = _ConvTransposeUpSampling


# class block(object):
#     # TODO: consistent API
#     OctConvBlock2d = _OctConvBlock2d

#     ConvBlock2d = _ConvBlock2d
#     ConvTransposeBlock2d = _ConvTransposeBlock2d
#     ResBlock2d = _ResBlock2d
#     ResTransposeBlock2d = _ResTransposeBlock2d
#     ResNextBlock2d = _ResNextBlock2d
#     ResNextTransposeBlock2d = _ResNextTransposeBlock2d
#     DenseBlock2d = _DenseBlock2d
#     DenseTransposeBlock2d = _DenseTransposeBlock2d
#     AttentionBlock2d = _AttentionBlock2d
#     AttentionTransposeBlock2d = _AttentionTransposeBlock2d
#     Block = _Block
