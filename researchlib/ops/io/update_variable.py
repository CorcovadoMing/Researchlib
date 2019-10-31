from torch import nn
from ...utils import ParameterManager


class _UpdateVariable(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.var_key = key
    
    def forward(self, x):
        ParameterManager.set_variable(self.var_key, x)
        return x