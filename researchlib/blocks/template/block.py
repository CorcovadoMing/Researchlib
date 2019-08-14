from torch import nn

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
    
    def _get_norm_layer(self, norm_type):
        # TODO
        if self.preact:
            return nn.BatchNorm2d(self.in_dim)
        else:
            return nn.BatchNorm2d(self.out_dim)
    
    def _get_pool_layer(self, pool_type, pool_factor):
        # TODO
        return nn.MaxPool2d(2)
        
    def forward(self, x):
        pass