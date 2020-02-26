from ...loss import *
from ...ops import *
from ...models import *
from ...blocks import *


def _DAD(input_node, output_node='out', model_node='model'):
    model = [
        Node('x1', op.ActiveNoise(['mul', 'add'], 1, 2), input_node),
        Node('x2', op.ActiveNoise(['mul', 'add'], 1, 2), input_node),
    
    
        Node(model_node,
             op.Sequential(
                 AutoEncDec(1, 7, 
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
                 op.InstanceNorm2d(32),
             ),
            'x1'),


        Node(output_node, 
             op.Sequential(
                 op.ActiveNoise(['add'], 1, 2),
                 AutoConvNet(32, 4, 
                            stem=None,
                            type='vgg', 
                            filters=(32, -1),
                            norm_type='instance',
                            preact=False,
                            pool_freq=100,
                            act_type='leaky_relu'),
                op.Conv2d(32, 1, 1, bias=False),
            ), 
             model_node),

        Node('y_path', ['model'], 'x2'),
        Node('y_result', op.Name(detach=True), 'y_path'),

        Optimize('mae1', Loss.AdaptiveRobust(1*256*256), output_node, input_node),
        Optimize('mae2', Loss.AdaptiveRobust(32*256*256), 'y_result', model_node),
    ]
    return model
