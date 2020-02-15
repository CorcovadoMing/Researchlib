from torch.distributions.categorical import Categorical
from torch import nn


class _Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return Categorical(logits=x)