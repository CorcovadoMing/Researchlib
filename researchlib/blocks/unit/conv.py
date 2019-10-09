from ..template.block import _Block
from ...layers import layer
from torch import nn
import torch.nn.utils.spectral_norm as sn
from .norm_act import _norm_act

class _conv(_Block):
    def __postinit__(self):
        # Parameters
        prepool = self._get_param('prepool', False)
        pool_type = self._get_param('pool_type', 'Upsample') if self._is_transpose() else self._get_param('pool_type', 'MaxPool')
        pool_factor = self._get_param('pool_factor', 2)
        non_local = self._get_param('non_local', False) and self._get_param('kernel_size', 3) != 1
        non_local_type = self._get_param('non_local_type', 'EmbeddedGaussian')
        non_local_layer = layer.__dict__[non_local_type + 'NonLocalBlock' + self._get_dim_type()](self.in_dim) if non_local else None
        spectral_norm = self._get_param('sn', False)
        erased_activator = self._get_param('erased_activator', False)
        
        conv_kwargs = self._get_conv_kwargs()

        # Layers
        conv_layer = self.op(self.in_dim, self.out_dim, **conv_kwargs)
        norm_act_layer = _norm_act(self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact, **self._get_custom_kwargs())
        if spectral_norm: conv_layer = sn(conv_layer)
        pool_layer = self._get_pool_layer(pool_type, pool_factor, self.out_dim) if self.do_pool else None

        if self.preact:
            self.layers = [norm_act_layer, non_local_layer, conv_layer, pool_layer]
        else:
            if prepool:
                self.layers = [non_local_layer, conv_layer, pool_layer, norm_act_layer]
            else:
                self.layers = [non_local_layer, conv_layer, norm_act_layer, pool_layer]
        self.layers = nn.Sequential(*list(filter(None, self.layers)))

    def forward(self, x):
        return self.layers(x)
