import torch
from torch import nn


class _ConditionProjection(nn.Module):

    def __init__(self, condition_f, data_f=None):
        super().__init__()
        self.condition_f = condition_f
        self.data_f_flag = True if data_f is not None else False
        if self.data_f_flag:
            self.data_f = data_f

    def forward(self, x):
        data, condition = x
        if self.data_f_flag:
            data = self.data_f(data)
        return torch.cat([data, self.condition_f(condition)], dim=1)
