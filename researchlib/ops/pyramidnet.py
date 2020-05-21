from torch import nn
import math


class _PyramidNet110(nn.Module):
    def __init__(self, head=None, in_dim=3, **kwargs):
        super().__init__()
        from ..models import AutoConvNet, Heads
        from ..blocks import unit
        from . import op
        default_kwargs = dict(
            _op = op.Conv2d,
            _unit = unit.Conv,
            input_dim = in_dim,
            total_blocks = 54,
            type='residual', 
            filters=(16, -1), 
            pool_freq = [18, 36],
            preact=True,
            erased_act = True,
            preact_final_norm = True,
            filter_policy = 'pyramid',
            pyramid_alpha = 200,
            shortcut = 'padding'
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg', preact=True))
        self.net = nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)


class _PyramidNet272(nn.Module):
    def __init__(self, head=None, in_dim=3, **kwargs):
        super().__init__()
        from ..models import AutoConvNet, Heads
        from ..blocks import unit
        from . import op
        default_kwargs = dict(
            _op = op.Conv2d,
            _unit = unit.Conv,
            input_dim = in_dim,
            total_blocks = 90,
            type='residual-bottleneck', 
            filters=(16, -1), 
            pool_freq = [30, 60],
            preact=True,
            erased_act = True,
            preact_final_norm = True,
            filter_policy = 'pyramid',
            pyramid_alpha = 200,
            shortcut = 'padding'
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg', preact=True))
        self.net = nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)
