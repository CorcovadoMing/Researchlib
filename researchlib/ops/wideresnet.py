from torch import nn


class _WideResNet28x10(nn.Module):
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
            total_blocks = 12,
            type='wide-residual', 
            filters=(16, -1), 
            pool_freq=[5, 9], 
            preact=True
        )
        default_kwargs.update(kwargs)
        model = [AutoConvNet(**default_kwargs)]
        if head != None:
            model.append(Heads(head, reduce_type='avg', preact=True))
        self.net = nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)