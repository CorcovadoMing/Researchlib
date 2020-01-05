from torch import nn
import torch
import torch.nn.functional as F


class _SConv2d(nn.Module):
    '''
        Reference Paper: Learning Implicitly Recurrent CNNs Through Parameter Sharing
    '''
    def __init__(self, bank, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.coefficients = nn.Parameter(torch.zeros(bank.coefficient_shape))

    def forward(self, input):
        params = self.bank(self.coefficients)
        return F.conv2d(input, params, stride=self.stride, padding=self.padding)