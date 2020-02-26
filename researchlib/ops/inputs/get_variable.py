from torch import nn
from ...utils import ParameterManager

class _GetVariable(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.var_key = key
    
    def forward(self, x):
        return ParameterManager.get_variable(self.var_key)