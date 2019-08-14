from .template.block import _Block
from ..layers import layer
from torch import nn

class ConvBlock(_Block):
    def __postinit__(self):
        kernel_size = self._get_param('kernel_size', 3)
        stride = self._get_param('stride', 1)
        padding = self._get_param('padding', 1)
        groups = self._get_param('groups', 1)
        dilation = self._get_param('dilation', 1)
        bias = self._get_param('bias', False)
        preact = self._get_param('preact', False)
        norm_type = self._get_param('norm_type', 'batchnorm')
        pool_type = self._get_param('pool_type', 'maxpool')
        pool_factor = self._get_param('pool_factor', 2)
        
        
        # Layers
        conv_layer = self.op(self.in_dim, self.out_dim, kernel_size, stride, padding, dilation, groups, bias)
        activator_layer = nn.ReLU() # TODO
        pool_layer = self._get_pool_layer(preact, pool_type, pool_factor) if self.do_pool else None
        norm_layer = self._get_norm_layer(preact, norm_type) if self.do_norm else None
        
        if preact:
            self.layers = [norm_layer, activator_layer, conv_layer, pool_layer]
        else:
            self.layers = [conv_layer, norm_layer, activator_layer, pool_layer]
        
        self.layers = nn.Sequential(*list(filter(None, self.layers)))
        
        
    def forward(self, x):
        return self.layers(x)