from ...loss import *
from ...ops import *
from ...models import *
from ...blocks import *


def _N2V(input_node, output_node='out', model_node='model'):
    model = [
        Node('x_patch', op.N2V.Patch(64, 16), input_node),
        Node('x_patch_and_mask', op.N2V.MaskAndReplace(16), 'x_patch'),
    
        Node(model_node,
             nn.Sequential(
                 AutoEncDec(op.Conv2d, 
                            op.ConvTranspose2d, 
                            unit.Conv, 1, 7, 
                            down_type='residual',
                            up_type='vgg',
                            preact=False, 
                            filters=(32, 512),
                            skip_type='concat',
                            return_bottleneck=False,
                            act_type='leaky_relu',
                            norm_type='instance',
                            blur=False,
                            pool_freq=2),
                 nn.InstanceNorm2d(32),
                 AutoConvNet(op.Conv2d, unit.Conv, 32, 
                           4, 
                           stem=None,
                           type='vgg', 
                           filters=(32, -1),
                           norm_type='instance',
                           preact=False,
                           pool_freq=100,
                           act_type='leaky_relu'),
                 nn.Conv2d(32, 1, 1, bias=False),
             ),
            'x_patch_and_mask:0'),
        
        Node(output_node, op.Name(), model_node),

        Node('masked_x', op.N2V.LossMask(), model_node, 'x_patch_and_mask:1'),
        Node('masked_y', op.N2V.LossMask(), 'x_patch', 'x_patch_and_mask:1'),
        Optimize('mae1', Loss.L2(), 'masked_x', 'masked_y')
    ]
    
    return model