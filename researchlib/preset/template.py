from ..models import AutoConvNet
from ..ops import op
from ..blocks import unit


class Template(object):
    Dawnfast = lambda: AutoConvNet(op.Conv2d, unit.Conv, 3, 3, stem={'whitening': 1},
                           type={'order':['dawn', 'vgg'], 'type':'alternative'}, 
                           filters=(64, 512), act_type='celu', prepool=True, freeze_scale=True, norm_type=op.GhostBatchNorm2d),
    
    ResNet18 = lambda: AutoConvNet(op.Conv2d, unit.Conv, 3, 8, stem={'whitening': 1}, type='residual', pool_freq=[3,5,7], filters=(64, -1))