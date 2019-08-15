from .template.block import _Block
from ..layers import layer
from torch import nn
from ._convblock import ConvBlock

class ResBlock(_Block):
    def __postinit__(self):
        self.conv = ConvBlock(self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact, **self.kwargs)
        
        pool_type = self._get_param('pool_type', 'MaxPool')
        pool_factor = self._get_param('pool_factor', 2)
        pool_layer = self._get_pool_layer(pool_type, pool_factor) if self.do_pool else None
          
        if self.in_dim != self.out_dim:
            reduction_op = layer.__dict__['Conv'+self._get_dim_type()](self.in_dim, self.out_dim, 1)
        else:
            reduction_op = None
            
        self.shortcut = nn.Sequential(*list(filter(None, [reduction_op, pool_layer])))
        
        
        # Se
        self.se = self._get_param('se', True)
        self.se_branch = nn.Sequential(
            layer.__dict__['AdaptiveMaxPool'+self._get_dim_type()](1),
            self.op(self.out_dim, self.out_dim // 16, kernel_size=1),
            nn.ReLU(),
            self.op(self.out_dim // 16, self.out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # shakedrop (sd)
        self.sd = self._get_param('sd', False)
        if self.sd:
            self.block_idx = self._get_param('block_idx', required=True)
            self.block_num = self._get_param('block_num', required=True)
            self.alpha_range = self._get_param('alpha_range', init_value=[-1, 1])
            self.beta_range = self._get_param('beta_range', init_value=[0, 1])
            self.shakedrop = layer.ShakeDrop(self.block_idx,
                                             self.block_num,
                                             p=0.5,
                                             alpha_range=self.alpha_range,
                                             beta_range=self.beta_range,
                                             p_L=0.5)
        
        
    def forward(self, x):
        _x = self.conv(x)
        if self.se:
            _x = self.se_branch(_x)
        if self.sd:
            _x = self.shakedrop(_x)
        x = self.shortcut(x)
        return x + _x