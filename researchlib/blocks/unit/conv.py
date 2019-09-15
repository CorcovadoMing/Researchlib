from ..template.block import _Block
from ...layers import layer
from torch import nn
import torch.nn.utils.spectral_norm as sn


class _conv(_Block):
    '''
        EraseReLU: A Simple Way to Ease the Training of Deep Convolution Neural Networks
        https://arxiv.org/pdf/1709.07634
    '''

    def __postinit__(self):
        # Parameters
        activator_type = self._get_param('activator_type', 'ReLU')
        norm_type = self._get_param('norm_type', 'BatchNorm')
        pool_type = self._get_param('pool_type', 'Upsample') if self._is_transpose() else self._get_param('pool_type', 'MaxPool')
        pool_factor = self._get_param('pool_factor', 2)
        non_local = self._get_param(
            'non_local', False) and self._get_param('kernel_size', 3) != 1
        non_local_type = self._get_param('non_local_type', 'EmbeddedGaussian')
        non_local_layer = layer.__dict__[non_local_type + 'NonLocalBlock' +
                                         self._get_dim_type()](
                                             self.in_dim) if non_local else None
        conv_kwargs = self._get_conv_kwargs()

        spectral_norm = self._get_param('sn', False)
        erased_activator = self._get_param('erased_activator', False)

        # Layers
        conv_layer = self.op(self.in_dim, self.out_dim, **conv_kwargs)
        if spectral_norm:
            conv_layer = sn(conv_layer)
        activator_layer = self._get_activator_layer(
            activator_type) if not erased_activator else None
        pool_layer = self._get_pool_layer(
            pool_type, pool_factor, self.out_dim) if self.do_pool else None
        norm_layer = self._get_norm_layer(norm_type) if self.do_norm else None

        if self.preact:
            self.layers = [
                norm_layer, activator_layer, non_local_layer, conv_layer,
                pool_layer
            ]
        else:
            self.layers = [
                non_local_layer, conv_layer, norm_layer, activator_layer,
                pool_layer
            ]
        self.layers = nn.Sequential(*list(filter(None, self.layers)))

    def forward(self, x):
        return self.layers(x)
