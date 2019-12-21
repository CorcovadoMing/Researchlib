from torch import nn
import torch.nn.functional as F


class _TReLU(nn.Module):
    def __init__(self, threshold= - .25, mean_shift=-.03):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift
    
    def forward(self,x):
        x = F.relu(x) + self.threshold
        x.sub_(self.mean_shift)    
        return x   