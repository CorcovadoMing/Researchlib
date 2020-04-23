from torch import nn


class _ResNet18(nn.Module):
    '''
        Deep Residual Learning for Image Recognition
    '''
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
            type='residual', 
            pool_freq=[3,5,7],
            filters=(64, -1),
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg'))
        self.net = nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)


class _ResNet50(nn.Module):
    def __init__(self, head=None, in_dim=3, **kwargs):
        super().__init__()
        from ..models import AutoConvNet, Heads
        from ..blocks import unit
        from . import op
        
        default_kwargs = dict(
            _op = op.Conv2d,
            _unit = unit.Conv,
            input_dim = in_dim,
            total_blocks = 16,
            stem={'vgg': 1}, 
            type='residual-bottleneck', 
            pool_freq=[4,8,14],
            filters=(256, -1)
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg'))
        self.net = nn.Sequential(*model)
        
    def forward(self, x):
        return self.net(x)