from .template.block import _Block
from ..layers import layer
from torch import nn
import torch
from .unit import unit

class InceptionA(_Block):
    '''
        Inception module with 4 pathways, only 1x1, 5x5, 3x3 kernel size being used
        AvgPool for the pool branch
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        self.branch1x1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        
        self.branch5x5_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False}))
        self.branch5x5_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 5, 'stride': stride, 'padding': 2, 'erased_activator': False}))
        
        self.branch3x3dbl_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False}))
        self.branch3x3dbl_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': 1, 'padding': 1, 'erased_activator': False}))
        self.branch3x3dbl_3 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': stride, 'padding': 1, 'erased_activator': False}))
        
        self.pool_fn = layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1)
        self.branch_pool = unit_fn(self.op, self.in_dim, remain_hidden, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.pool_fn(x)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(_Block):
    '''
        Inception module with 3 pathways, only 1x1, 3x3 kernel size being used
        MaxPool for the pool branch
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 3
        remain_hidden = self.out_dim - (2 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        self.branch3x3 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': stride, 'padding': 1, 'erased_activator': False}))
        
        self.branch3x3dbl_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False}))
        self.branch3x3dbl_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': 1, 'padding': 1, 'erased_activator': False}))
        self.branch3x3dbl_3 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': stride, 'padding': 1, 'erased_activator': False}))
        
        self.pool_fn = layer.__dict__['MaxPool' + self._get_dim_type()](3, 1, 1)
        self.branch_pool = unit_fn(self.op, self.in_dim, remain_hidden, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.pool_fn(x)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(_Block):
    '''
        Inception module with 3 pathways, only 1x1, 1x3, 7x1, 1x7 kernel size being used
        7x7 kernel path is with bottleneck style
        AvgPool for the pool branch
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        self.branch1x1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        
        self.branch7x7_1 = unit_fn(self.op, self.in_dim, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        self.branch7x7_2 = unit_fn(self.op, hidden_dim // 2, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (1, 7), 'stride': 1, 'padding': (0, 3), 'erased_activator': False}))
        self.branch7x7_3 = unit_fn(self.op, hidden_dim // 2, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (7, 1), 'stride': 1, 'padding': (3, 0), 'erased_activator': False}))
        
        self.branch7x7dbl_1 = unit_fn(self.op, self.in_dim, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        self.branch7x7dbl_2 = unit_fn(self.op, hidden_dim // 2, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (7, 1), 'stride': 1, 'padding': (3, 0), 'erased_activator': False}))
        self.branch7x7dbl_3 = unit_fn(self.op, hidden_dim // 2, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (1, 7), 'stride': 1, 'padding': (0, 3), 'erased_activator': False}))
        self.branch7x7dbl_4 = unit_fn(self.op, hidden_dim // 2, hidden_dim // 2, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (7, 1), 'stride': 1, 'padding': (3, 0), 'erased_activator': False}))
        self.branch7x7dbl_5 = unit_fn(self.op, hidden_dim // 2, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (1, 7), 'stride': 1, 'padding': (0, 3), 'erased_activator': False}))
        
        self.pool_fn = layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1)
        self.branch_pool = unit_fn(self.op, self.in_dim, remain_hidden, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        
        branch_pool = self.pool_fn(x)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(_Block):
    '''
        Inception module with 3 pathways, only 1x1, 3x3, 1x7, 7x1 kernel size being used
        MaxPool for the pool branch
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 3
        remain_hidden = self.out_dim - (2 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        self.branch3x3_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False}))
        self.branch3x3_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': stride, 'padding': 1, 'erased_activator': False}))
        
        self.branch7x7x3_1 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': 1, 'padding': 0, 'erased_activator': False}))
        self.branch7x7x3_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (1, 7), 'stride': 1, 'padding': (0, 3), 'erased_activator': False}))
        self.branch7x7x3_3 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': (7, 1), 'stride': 1, 'padding': (3, 0), 'erased_activator': False}))
        self.branch7x7x3_4 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 3, 'stride': stride, 'padding': 1, 'erased_activator': False}))
        
        self.pool_fn = layer.__dict__['MaxPool' + self._get_dim_type()](3, 1, 1)
        self.branch_pool = unit_fn(self.op, self.in_dim, remain_hidden, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        
        branch_pool = self.pool_fn(x)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(_Block):
    '''
        Inception module with 4 pathways, only 1x1, 1x3, 3x1, 3x3 kernel size being used
        AvgPool for the pool branch
    '''
    def __postinit__(self):
        unit_fn = self._get_param('unit', unit.conv)
        hidden_dim = self.out_dim // 4
        remain_hidden = self.out_dim - (3 * hidden_dim)
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        self.branch1x1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                 **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        
        self.branch3x3_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        hidden_dim_2a = hidden_dim // 2
        hidden_dim_2b = hidden_dim - hidden_dim_2a
        self.branch3x3_2a = unit_fn(self.op, hidden_dim, hidden_dim_2a, False, True, False,
                                    **self._get_custom_kwargs({'kernel_size': (1, 3), 'stride': 1, 'padding': (0, 1), 'erased_activator': False}))
        self.branch3x3_2b = unit_fn(self.op, hidden_dim, hidden_dim_2b, False, True, False,
                                    **self._get_custom_kwargs({'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0), 'erased_activator': False}))
        
        self.branch3x3dbl_1 = unit_fn(self.op, self.in_dim, hidden_dim, False, True, False,
                                      **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))
        self.branch3x3dbl_2 = unit_fn(self.op, hidden_dim, hidden_dim, False, True, False,
                                      **self._get_custom_kwargs({'kernel_size': 3, 'stride': 1, 'padding': 1, 'erased_activator': False}))
        hidden_dim_3a = hidden_dim // 2
        hidden_dim_3b = hidden_dim - hidden_dim_3a
        self.branch3x3dbl_3a = unit_fn(self.op, hidden_dim, hidden_dim_3a, False, True, False,
                                       **self._get_custom_kwargs({'kernel_size': (1, 3), 'stride': 1, 'padding': (0, 1), 'erased_activator': False}))
        self.branch3x3dbl_3b = unit_fn(self.op, hidden_dim, hidden_dim_3b, False, True, False,
                                       **self._get_custom_kwargs({'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0), 'erased_activator': False}))
        
        self.pool_fn = layer.__dict__['AvgPool' + self._get_dim_type()](3, 1, 1)
        self.branch_pool = unit_fn(self.op, self.in_dim, remain_hidden, False, True, False,
                                   **self._get_custom_kwargs({'kernel_size': 1, 'stride': stride, 'padding': 0, 'erased_activator': False}))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        
        branch_pool = self.pool_fn(x)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)