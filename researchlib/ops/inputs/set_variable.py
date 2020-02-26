from torch import nn
from ...utils import ParameterManager

class _SetVariable(nn.Module):
    def __init__(self, key, value):
        super().__init__()
        ParameterManager.set_variable(key, value)
        self.var_key = key
    
    def forward(self, x):
        return ParameterManager.get_variable(self.var_key)