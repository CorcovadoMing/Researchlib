from torch import nn
        
def _is_container(m):
    from ..models import GANModel
    from ..layers import Auxiliary
    return isinstance(m, nn.Sequential)  \
            or isinstance(m, GANModel)   \
            or isinstance(m, Auxiliary)