from .template.block import _Block
from .tcnblock import _TCNBlock
from .gcnblock import _GCNBlock
from torch import nn


class _AGCBlock(_Block):
    def __postinit__(self):
        stride = self._get_param('stride', 1)
        self.gcn = _GCNBlock(
            self.op, self.in_dim, self.out_dim, False, True, False, **self._get_custom_kwargs()
        )
        self.tcn = _TCNBlock(
            self.op, self.in_dim, self.out_dim, False, True, False, **self._get_custom_kwargs()
        )
        self.relu = nn.ReLU()

        if (self.in_dim == self.out_dim) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = self.op(self.in_dim, self.out_dim, 1, stride = stride)

    def forward(self, x):
        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)
