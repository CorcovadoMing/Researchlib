from torch import nn
import math


# class _ResNet(nn.Module):
#     '''
#         Deep Residual Learning for Image Recognition
#     '''
#     def __init__(self, size, pool_freq=2, head=None, in_dim=3, **kwargs):
#         super().__init__()
#         from ..models import AutoConvNet, Heads
#         from ..blocks import unit
#         from . import op
        
#         use_large_stem = True if size >= 64 else False
#         total_blocks = (int((math.log2(size)-3+1) - (2 if use_large_stem else 0)) * pool_freq)
#         _pool_freq = [i+pool_freq+1 for i in range(0, total_blocks, pool_freq)]
#         default_kwargs = dict(
#             _op = op.Conv2d,
#             _unit = unit.Conv,
#             input_dim = in_dim,
#             total_blocks = total_blocks,
#             stem={'vgg': 1}, 
#             stem_large=use_large_stem,
#             type='residual', 
#             pool_freq=_pool_freq,
#             filters=(64, -1),
#         )
#         default_kwargs.update(kwargs)
#         model = [AutoConvNet(**default_kwargs)]
#         if head != None:
#             model.append(Heads(head, reduce_type='avg'))
#         self.net = nn.Sequential(*model)
            
#     def forward(self, x):
#         return self.net(x)


class _VGG19(nn.Module):
    def __init__(self, head=None, in_dim=3, **kwargs):
        super().__init__()
        from ..models import AutoConvNet, Heads
        from ..blocks import unit
        from . import op
        
        default_kwargs = dict(
            _op = op.Conv2d,
            _unit = unit.Conv,
            input_dim = in_dim,
            total_blocks = 8,
            stem={'vgg': 1}, 
            type='vgg', 
            pool_freq=[3,5,8],
            filters=(64, -1),
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg'))
        self.net = nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)
