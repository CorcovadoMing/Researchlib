from .template.block import _Block
from ..layers import layer
from torch import nn
import torch.nn.functional as F
import torch
from .unit import unit


class _InceptionResidualV2A(_Block):

    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 3
        remain_hidden = self.out_dim - (2 * hidden_dim)

        self.scale = 1.0

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.branch2 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.conv2d = unit_fn(
            self.op, self.out_dim, self.out_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class _ReductionV2A(_Block):

    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 3
        remain_hidden = self.out_dim - (2 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 3,
                'stride': stride,
                'padding': 1,
                'erased_activator': False
            }))

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.branch2 = nn.Sequential(
            layer.__dict__['MaxPool' + self._get_dim_type()](3, stride, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class _InceptionResidualV2B(_Block):

    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 2
        remain_hidden = self.out_dim - (1 * hidden_dim)

        self.scale = 1.0

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 7),
                    'stride': 1,
                    'padding': (0, 3),
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (7, 1),
                    'stride': 1,
                    'padding': (3, 0),
                    'erased_activator': False
                })),
        )

        self.conv2d = unit_fn(
            self.op, self.out_dim, self.out_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class _ReductionV2B(_Block):

    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1

        self.branch0 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.branch2 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })))

        self.branch3 = nn.Sequential(
            layer.__dict__['MaxPool' + self._get_dim_type()](3, stride, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class _InceptionResidualV2C(_Block):

    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 2
        remain_hidden = self.out_dim - (1 * hidden_dim)

        self.scale = 1.0

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 3),
                    'stride': 1,
                    'padding': (0, 1),
                    'erased_activator': False
                })),
            unit_fn(
                self.op, remain_hidden, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (3, 1),
                    'stride': 1,
                    'padding': (1, 0),
                    'erased_activator': False
                })))

        self.conv2d = unit_fn(
            self.op, self.out_dim, self.out_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            }))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
