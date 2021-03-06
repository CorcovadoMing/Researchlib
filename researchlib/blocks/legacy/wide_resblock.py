from .template.block import _Block
from torch import nn
from .unit import unit


class _WideResBlock(_Block):
    '''
        TODO
    '''
    def __postinit__(self):
        is_transpose = self._is_transpose()

        unit_fn = self._get_param('unit', unit.conv)
        erased_activator = self._get_param('erased_activator', False)
        activator_type = self._get_param('activator_type', 'ReLU')
        activator_layer = self._get_activator_layer(
            activator_type
        ) if not erased_activator and not self.preact else None
        self.merge_layer = nn.Sequential(*list(filter(None, [activator_layer])))

        norm_type = self._get_param('norm_type', 'BatchNorm')
        preact_final_norm_layer = self._get_norm_layer(
            norm_type, self.out_dim
        ) if self.do_norm and self.preact else None

        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        kernel_size = 2 if is_transpose and self.do_pool else self._get_param('kernel_size', 3)
        padding = 0 if is_transpose and self.do_pool else self._get_param(
            'padding', int((kernel_size - 1) / 2)
        )
        drop_layer = nn.Dropout(0.5) if self._get_param('dropout', True) else None
        first_custom_kwargs = self._get_custom_kwargs({
            'kernel_size':
            kernel_size,
            'stride':
            stride,
            'padding':
            padding,
            'erased_activator':
            True if self.preact and erased_activator else False
        })
        second_custom_kwargs = self._get_custom_kwargs({
            'erased_activator':
            True if not self.preact else False
        })

        conv_layers = [
            unit_fn(
                self.op, self.in_dim, self.out_dim, False, self.do_norm, self.preact,
                **first_custom_kwargs
            ), drop_layer,
            unit_fn(
                self.op, self.out_dim, self.out_dim, False, self.do_norm, self.preact,
                **second_custom_kwargs
            ), preact_final_norm_layer
        ]
        self.conv = nn.Sequential(*list(filter(None, conv_layers)))

        self.shortcut = self._get_shortcut()

        self.branch_attention = self._get_param('branch_attention')
        if self.branch_attention:
            self.attention_branch = self._get_attention_branch()
        self.shakedrop = self._get_param('shakedrop', False)
        if self.shakedrop:
            self.shakedrop_branch = self._get_shake_drop_branch()

    def forward(self, x):
        _x = self.conv(x)
        if self.branch_attention:
            _x = self.attention_branch(_x)
        if self.shakedrop:
            _x = self.shakedrop_branch(_x)
        x = self.shortcut(x)
        return self.merge_layer(x + _x)
