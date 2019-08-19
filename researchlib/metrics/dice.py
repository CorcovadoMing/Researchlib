from .matrix import *
from ..utils import *

import numpy as np
import matplotlib.pyplot as plt
import torch


class DiceMatrix(Matrix):

    def __init__(self, smooth=1, target_class=1, need_exp=False):
        super().__init__()
        self.smooth = smooth
        self.target_class = target_class
        self.need_exp = need_exp
        self.correct = 0
        self.total = 0

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup (INCOMPLETE!!)
            pass
        else:
            y_pred, y_true = loss_input[0], loss_input[1]
            y_pred, y_true = y_pred.detach(), y_true.detach()
            y_pred = y_pred[:, self.target_class, :, :]
            if self.need_exp:
                y_pred = y_pred.exp()
            y_pred = y_pred.view(y_pred.size(0), -1).contiguous()
            y_true = (y_true == self.target_class)
            y_true = y_true.view(y_pred.size(0), -1).float().contiguous()
            intersection = (y_pred * y_true).sum(1)
            ratio = 2 * (intersection + self.smooth) / (
                y_true.sum(1) + y_pred.sum(1) + self.smooth)
            self.correct += ratio.mean()
        self.total += 1

    def output(self):
        dice = self.correct / self.total
        return {'dice': dice}

    def reset(self):
        self.correct = 0
        self.total = 0
