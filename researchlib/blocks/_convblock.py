from .template.block import _Block
from ..layers import layer
from torch import nn
import torch.nn.utils.spectral_norm as sn


class ConvBlock(_Block):
    def __postinit__(self):
        # Parameters
        norm_type = self._get_param('norm_type', 'BatchNorm')
        pool_type = self._get_param('pool_type', 'MaxPool')
        pool_factor = self._get_param('pool_factor', 2)
        conv_kwargs = self._get_conv_kwargs()

        # Layers
        spectral_norm = self._get_param('sn', False)
        conv_layer = self.op(self.in_dim, self.out_dim, **conv_kwargs)
        if spectral_norm:
            conv_layer = sn(conv_layer)
        activator_layer = nn.ReLU()  # TODO
        pool_layer = self._get_pool_layer(
            pool_type, pool_factor) if self.do_pool else None
        norm_layer = self._get_norm_layer(norm_type) if self.do_norm else None

        if self.preact:
            self.layers = [norm_layer, activator_layer, conv_layer, pool_layer]
        else:
            self.layers = [conv_layer, norm_layer, activator_layer, pool_layer]
        self.layers = nn.Sequential(*list(filter(None, self.layers)))

    def forward(self, x):
        return self.layers(x)
