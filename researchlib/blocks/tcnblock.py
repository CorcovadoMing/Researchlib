from .template.block import _Block
from .unit import unit


class _TCNBlock(_Block):

    def __postinit__(self):
        # TODO (Ming): Maybe useless, but keep here if the gereative model is used in future
        is_transpose = self._is_transpose()

        custom_kwargs = self._get_custom_kwargs()

        # TODO (Ming): Only 2d supported
        custom_kwargs['kernel_size'] = (self._get_param('tcn_kernel_size',
                                                        9), 1)
        custom_kwargs['padding'] = ((custom_kwargs['kernel_size'][0] - 1) / 2,
                                    0)
        custom_kwargs['stride'] = (self._get_param('tcn_stride', 1), 1)

        unit_fn = self._get_param('unit', unit.conv)
        # TODO (Ming): do_pool=False, preact=False for now
        self.unit = unit_fn(self.op, self.in_dim, self.out_dim, False,
                            self.do_norm, False, **custom_kwargs)

    def forward(self, x):
        return self.unit(x)
