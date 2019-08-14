from torch import nn

class _Block(nn.Module):
    def __init__(self, op, in_dim, out_dim, do_pool, do_norm, **kwargs):
        super().__init__()
        self.op = op
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.do_pool = do_pool
        self.do_norm = do_norm
        self.kwargs = kwargs
        self.__postinit__()
    
    def __postinit__(self):
        pass
    
    def _get_param(self, key, init_value):
        try:
            query = self.kwargs[key]
            return query
        except:
            return init_value
    
    def _get_norm_layer(self, preact, norm_type):
        if preact:
            return nn.BatchNorm2d(self.in_dim)
        else:
            return nn.BatchNorm2d(self.out_dim)
    
    def _get_pool_layer(self, preact, pool_type, pool_factor):
        return nn.MaxPool2d(2)
        
    def forward(self, x):
        pass