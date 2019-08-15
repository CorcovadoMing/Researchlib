import re 
import torch
from torch import nn
from ...layers import layer

class _Block(nn.Module):
    def __init__(self, op, in_dim, out_dim, do_pool, do_norm, preact, **kwargs):
        super().__init__()
        self.op = op
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.do_pool = do_pool
        self.do_norm = do_norm
        self.preact = preact
        self.kwargs = kwargs
        
        # Customize
        self.__postinit__()
    
    def __postinit__(self):
        pass
    
    def _get_conv_args(self):
        kernel_size = self._get_param('kernel_size', 3)
        stride = self._get_param('stride', 1)
        padding = self._get_param('padding', 1)
        dilation = self._get_param('dilation', 1)
        groups = self._get_param('groups', 1)
        bias = self._get_param('bias', False)
        return kernel_size, stride, padding, dilation, groups, bias
    
    def _get_param(self, key, init_value):
        try:
            query = self.kwargs[key]
            return query
        except:
            return init_value
    
    def _get_dim_type(self):
        match = re.search('\dd', str(self.op))
        dim_str = match.group(0)
        return dim_str
    
    def _get_norm_layer(self, norm_type):
        if norm_type not in ['BatchNorm', 'InstanceNorm', 'GroupNorm']:
            raise('Unknown norm type')
        
        if self.preact:
            dim = [self.in_dim]
        else:
            dim = [self.out_dim]
        
        if norm_type is not 'GroupNorm':
            dim_str = self._get_dim_type()
        else:
            dim_str = ''
            group_num = self._get_param('groupnorm_group', 4)
            dim.insert(0, group_num)
        norm_op_str = norm_type + dim_str
        norm_op = layer.__dict__[norm_op_str]
        
        return norm_op(*dim)
    
    
    def _get_pool_layer(self, pool_type, pool_factor):
        if pool_type not in ['MaxPool', 'AvgPool', 'Combined']:
            raise('Unknown pool type')
        
        dim_str = self._get_dim_type()
        if pool_type is not 'Combined':
            pool_op_str = pool_type + dim_str
            pool_op = layer.__dict__[pool_op_str]
            return pool_op(pool_factor)
        else:
            max_pool_op = layer.__dict__['MaxPool'+dim_str](pool_factor)
            avg_pool_op = layer.__dict__['AvgPool'+dim_str](pool_factor)
            conv_pool_op = nn.Sequential(layer.__dict__['Conv'+dim_str](self.out_dim, self.out_dim, 4, 2, 1), nn.LeakyReLU(0.5)) # Special case
            reduction_op = layer.__dict__['Conv'+dim_str](self.out_dim*3, self.out_dim, 1)
            return _Combined([max_pool_op, avg_pool_op, conv_pool_op], reduction_op, self.preact)
        
    def forward(self, x):
        pass


# ================================================================================================
    
class _Combined(nn.Module):
    def __init__(self, fn_list, reduction_op, preact):
        super().__init__()
        self.fn_list = nn.ModuleList(fn_list)
        if preact:
            self.reduction_op = reduction_op
        else:
            self.reduction_op = nn.Sequential(reduction_op, nn.LeakyReLU(0.5))
        
    def forward(self, x):
        out = [f(x) for f in self.fn_list]
        out = torch.cat(out, dim=1)
        return self.reduction_op(out)