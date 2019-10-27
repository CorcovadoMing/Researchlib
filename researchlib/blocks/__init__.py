# from .octconvblock import _OctConvBlock2d
# from .convblock import _ConvBlock2d, _ConvTransposeBlock2d
# from .resblock import _ResBlock2d, _ResTransposeBlock2d
# from .resnextblock import _ResNextBlock2d, _ResNextTransposeBlock2d
# from .attentionblock import _AttentionBlock2d, _AttentionTransposeBlock2d
# from .denseblock import _DenseBlock2d, _DenseTransposeBlock2d
# from .basic_components import _CombinedDownSampling, _MaxPoolDownSampling, _InterpolateUpSampling, _ConvTransposeUpSampling

from .unit import unit
from .template.block import _Block as _TemplateBlock
from .dawnfast import _DAWNBlock
from .resblock_bottleneck import _ResBottleneckBlock
from .resblock import _ResBlock
from .vggblock import _VGGBlock
from .wide_resblock import _WideResBlock

from .tcnblock import _TCNBlock
from .gcnblock import _GCNBlock
from .agcblock import _AGCBlock

from .whitening_block import _WhiteningBlock


class block(object):
    TemplateBlock = _TemplateBlock

    WhiteningBlock = _WhiteningBlock
    
    AGCBlock = _AGCBlock
    TCNBlock = _TCNBlock
    GCNBlock = _GCNBlock
    DAWNBlock = _DAWNBlock

    ResBottleneckBlock = _ResBottleneckBlock
    ResBlock = _ResBlock
    WideResBlock = _WideResBlock
    
    VGGBlock = _VGGBlock