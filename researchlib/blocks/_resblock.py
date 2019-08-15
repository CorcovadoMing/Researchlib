from .template.block import _Block
from ..layers import layer
from torch import nn
from ._convblock import ConvBlock


class ResBlock(_Block):
    def __postinit__(self):
        is_transpose = self._is_transpose()
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        padding = 0 if is_transpose and self.do_pool else self._get_param('padding', 1)
        kernel_size = 2 if is_transpose and self.do_pool else self._get_param('kernel_size', 3)
        self.conv = nn.Sequential(
            ConvBlock(self.op, self.in_dim, self.out_dim, False, self.do_norm, self.preact, kernel_size=kernel_size, stride=stride, padding=padding, **self.kwargs),
            ConvBlock(self.op, self.out_dim, self.out_dim, False, self.do_norm, self.preact, **self.kwargs)
        )
        
        shortcut_kernel_size = 2 if is_transpose and self.do_pool else 1
        if self.in_dim != self.out_dim or self.do_pool:
            reduction_op = self.op(self.in_dim, self.out_dim, kernel_size=shortcut_kernel_size, stride=stride)
        else:
            reduction_op = None 
        self.shortcut = nn.Sequential(*list(filter(None, [reduction_op])))

        # Se
        self.se = self._get_param('se', True)
        self.se_branch = nn.Sequential(
            layer.__dict__['AdaptiveMaxPool' + self._get_dim_type()](1),
            self.op(self.out_dim, self.out_dim // 16,
                    kernel_size=1), nn.ReLU(),
            self.op(self.out_dim // 16, self.out_dim, kernel_size=1),
            nn.Sigmoid())

        # shakedrop (sd)
        self.sd = self._get_param('sd', False)
        if self.sd:
            self.block_idx = self._get_param('id', required=True)
            self.block_num = self._get_param('total_blocks', required=True)
            self.alpha_range = self._get_param('alpha_range',
                                               init_value=[-1, 1])
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
            _x = _x * self.se_branch(_x)
        if self.sd:
            _x = self.shakedrop(_x)
        x = self.shortcut(x)
        return x + _x
