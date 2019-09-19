from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit


class _InceptionV4A(_Block):
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            })
        )

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })
            )
        )

        self.branch2 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })
            )
        )

        self.branch3 = nn.Sequential(
            layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            )
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class _ReductionV4A(_Block):
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
            })
        )

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })
            ),
        )

        self.branch2 = nn.Sequential(
            layer.__dict__['MaxPool' + self._get_dim_type()](3, 2, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            )
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class _InceptionV4B(_Block):
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            })
        )

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 7),
                    'stride': 1,
                    'padding': (0, 3),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (7, 1),
                    'stride': 1,
                    'padding': (3, 0),
                    'erased_activator': False
                })
            )
        )

        self.branch2 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (7, 1),
                    'stride': 1,
                    'padding': (3, 0),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 7),
                    'stride': 1,
                    'padding': (0, 3),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (7, 1),
                    'stride': 1,
                    'padding': (3, 0),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 7),
                    'stride': 1,
                    'padding': (0, 3),
                    'erased_activator': False
                })
            )
        )

        self.branch3 = nn.Sequential(
            layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            )
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class _ReductionV4B(_Block):
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 3
        remain_hidden = self.out_dim - (2 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1

        self.branch0 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })
            )
        )

        self.branch1 = nn.Sequential(
            unit_fn(
                self.op, self.in_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (1, 7),
                    'stride': 1,
                    'padding': (0, 3),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': (7, 1),
                    'stride': 1,
                    'padding': (3, 0),
                    'erased_activator': False
                })
            ),
            unit_fn(
                self.op, hidden_dim, hidden_dim, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 3,
                    'stride': stride,
                    'padding': 1,
                    'erased_activator': False
                })
            )
        )

        self.branch2 = nn.Sequential(
            layer.__dict__['MaxPool' + self._get_dim_type()](3, 2, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            )
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class _InceptionV4C(_Block):
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)

        self.branch0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            })
        )

        self.branch1_0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            })
        )
        hidden_dim_1a = hidden_dim // 2
        hidden_dim_1b = hidden_dim - hidden_dim_1a
        self.branch1_1a = unit_fn(
            self.op, hidden_dim, hidden_dim_1a, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (1, 3),
                'stride': 1,
                'padding': (0, 1),
                'erased_activator': False
            })
        )
        self.branch1_1b = unit_fn(
            self.op, hidden_dim, hidden_dim_1b, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (3, 1),
                'stride': 1,
                'padding': (1, 0),
                'erased_activator': False
            })
        )

        self.branch2_0 = unit_fn(
            self.op, self.in_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'erased_activator': False
            })
        )
        self.branch2_1 = unit_fn(
            self.op, hidden_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (3, 1),
                'stride': 1,
                'padding': (1, 0),
                'erased_activator': False
            })
        )
        self.branch2_2 = unit_fn(
            self.op, hidden_dim, hidden_dim, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (1, 3),
                'stride': 1,
                'padding': (0, 1),
                'erased_activator': False
            })
        )
        hidden_dim_2a = hidden_dim // 2
        hidden_dim_2b = hidden_dim - hidden_dim_2a
        self.branch2_3a = unit_fn(
            self.op, hidden_dim, hidden_dim_2a, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (1, 3),
                'stride': 1,
                'padding': (0, 1),
                'erased_activator': False
            })
        )
        self.branch2_3b = unit_fn(
            self.op, hidden_dim, hidden_dim_2b, False, True, False,
            **self._get_custom_kwargs({
                'kernel_size': (3, 1),
                'stride': 1,
                'padding': (1, 0),
                'erased_activator': False
            })
        )

        self.branch3 = nn.Sequential(
            layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1),
            unit_fn(
                self.op, self.in_dim, remain_hidden, False, True, False,
                **self._get_custom_kwargs({
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'erased_activator': False
                })
            )
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out
