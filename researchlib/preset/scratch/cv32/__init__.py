from ....models import AutoConvNet, Heads, Node
from ....ops import op
from ....blocks import unit
from torch import nn

from .randwire import RandWire
from .dawnfast import Dawnfast
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet110
from .resnext import ResNeXt50
from .preresnet import PreResNet18, PreResNet34, PreResNet50, PreResNet101, PreResNet152
from .revnet import RevNet110
from .wideresnet import WideResNet28x10
from .pyramidnet import PyramidNet272



class CV32(object):
    RandWire = RandWire
    
    Dawnfast = Dawnfast
    
    ResNet18 = ResNet18
    ResNet34 = ResNet34
    ResNet50 = ResNet50
    ResNet101 = ResNet101
    ResNet152 = ResNet152
    
    ResNet110 = ResNet110
    
    ResNeXt50 = ResNeXt50
    
    RevNet110 = RevNet110
    
    PreResNet18 = PreResNet18
    PreResNet34 = PreResNet34
    PreResNet50 = PreResNet50
    PreResNet101 = PreResNet101
    PreResNet152 = PreResNet152
    
    WideResNet28x10 = WideResNet28x10
    
    PyramidNet272 = PyramidNet272
    