from torch.distributions.categorical import Categorical
from torch import nn
import torch.nn.functional as F


class _Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return Categorical(logits=x)