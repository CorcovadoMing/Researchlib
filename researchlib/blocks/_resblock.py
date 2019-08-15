from .template.block import _Block
from ..layers import layer
from torch import nn
from ._convblock import ConvBlock

class ResBlock(_Block):
    def __postinit__(self):
        self.conv = ConvBlock(self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact, **self.kwargs)
        if self.do_pool:
            reduction_stride = self._get_param('pool_factor', 2)
        else:
            reduction_stride = 1
            
        if self.in_dim != self.out_dim:
            self.do_reduction = True
            self.reduction_op = layer.__dict__['Conv'+self._get_dim_type()](self.in_dim, self.out_dim, 1, reduction_stride)
        else:
            self.do_reduction = False
        
        # Se
        self.se = self._get_param('se', True)
        self.se_branch = nn.Sequential(
            layer.__dict__['AdaptiveMaxPool'+self._get_dim_type()](1),
            self.op(self.out_dim, self.out_dim // 16, kernel_size=1),
            nn.ReLU(),
            self.op(self.out_dim // 16, self.out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        _x = self.conv(x)
        if self.se:
            _x = self.se_branch(_x)
        if self.do_reduction:
            x = self.reduction_op(x)
        return x + _x