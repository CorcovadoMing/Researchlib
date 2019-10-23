from .template.block import _Block
from .unit import unit


class _VGGBlock(_Block):
    '''
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        self.unit = unit_fn(
            self.op, self.in_dim, self.out_dim, self.do_pool, self.do_norm, self.preact,
            **self._get_custom_kwargs()
        )

    def forward(self, x):
        x = self.unit(x)
        return x
