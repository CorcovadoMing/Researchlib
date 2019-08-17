from .template.block import _Block
from ..layers import layer
from torch import nn
from ._convblock import ConvBlock
import copy

class ResBlock(_Block):
    '''
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    '''
    def __postinit__(self):
        is_transpose = self._is_transpose()

        erased_activator = self._get_param('erased_activator', False)
        activator_type = self._get_param('actvator_type', 'ReLU')
        activator_layer = self._get_activator_layer(activator_type) if not erased_activator and not self.preact else None
        self.merge_layer = nn.Sequential(*list(filter(None, [activator_layer])))

        # Layers
        total_blocks = self._get_param('total_blocks', required=True)
        # 16-resblocks equals to resnet 34, however, replace each 2-conv block with 3-conv block results in boost performance
        # Here replace the block design if using more than 15 resblocks, which means we don't have resnet-34, use resnet-50 instead in same total blocks.
        if total_blocks > 15:
            hidden_size = self.out_dim // 4
            stride = self._get_param('pool_factor', 2) if self.do_pool else 1
            padding = 0 if is_transpose and self.do_pool else self._get_param('padding', 1)
            kernel_size = 2 if is_transpose and self.do_pool else self._get_param('kernel_size', 3)
            first_custom_kwargs = self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False})
            second_custom_kwargs = self._get_custom_kwargs({'kenel_size': kernel_size, 'stride': stride, 'padding': padding, 'erased_activator': False})
            third_custom_kwargs = self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': True if not self.preact else False})
            self.conv = nn.Sequential(
                ConvBlock(self.op, self.in_dim, hidden_size, False, self.do_norm, self.preact, **first_custom_kwargs),
                ConvBlock(self.op, hidden_size, hidden_size, False, self.do_norm, self.preact, **second_custom_kwargs),
                ConvBlock(self.op, hidden_size, self.out_dim, False, self.do_norm, self.preact, **third_custom_kwargs)
            )
        else:
            stride = self._get_param('pool_factor', 2) if self.do_pool else 1
            padding = 0 if is_transpose and self.do_pool else self._get_param('padding', 1)
            kernel_size = 2 if is_transpose and self.do_pool else self._get_param('kernel_size', 3)
            first_custom_kwargs = self._get_custom_kwargs({'kenel_size': kernel_size, 'stride': stride, 'padding': padding, 'erased_activator': False})
            second_custom_kwargs = self._get_custom_kwargs({'erased_activator': True if not self.preact else False})
            self.conv = nn.Sequential(
                ConvBlock(self.op, self.in_dim, self.out_dim, False, self.do_norm, self.preact, **first_custom_kwargs),
                ConvBlock(self.op, self.out_dim, self.out_dim, False, self.do_norm, self.preact, **second_custom_kwargs)
            )
        
        shortcut_kernel_size = 2 if is_transpose and self.do_pool else 1
        if self.in_dim != self.out_dim or self.do_pool:
            custom_kwargs = self._get_custom_kwargs({'kenel_size': shortcut_kernel_size, 'stride': stride})
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
        return self.merge_layer(x + _x)
