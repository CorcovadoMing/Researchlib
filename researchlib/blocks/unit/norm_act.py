from ..template.block import _Block
from ...ops import op
from torch import nn


class _norm_act(_Block):
    def __postinit__(self):
        # Parameters
        activator_type = self._get_param('activator_type', 'ReLU')
        norm_type = self._get_param('norm_type', 'BatchNorm')
        erased_activator = self._get_param('erased_activator', False)

        # Layers
        activator_layer = self._get_activator_layer(
            activator_type
        ) if not erased_activator else None
        norm_layer = self._get_norm_layer(norm_type) if self.do_norm else None
        self.layers = nn.Sequential(*list(filter(None, [norm_layer, activator_layer])))

    def forward(self, x):
        return self.layers(x)
