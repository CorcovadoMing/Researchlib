from .matrix import *
from ..utils import *
import torch


class BCEAcc(Matrix):

    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup
            y_pred, y_true, y_true_res, lam = loss_input[0].detach(
            ), loss_input[1].detach(), loss_input[2].detach(), loss_input[3]
            predicted = (y_pred > 0.5).float()
            self.total += y_true.size(0)
            self.correct += (
                lam * predicted.eq(y_true).sum().float() +
                (1 - lam) * predicted.eq(y_true_res).sum().float())
        else:
            y_pred, y_true = loss_input[0].detach(), loss_input[1].detach(
            ).squeeze()
            predicted = (y_pred > 0.5).float()
            self.total += y_true.size(0)
            self.correct += predicted.eq(y_true).sum().float()

    def output(self):
        acc = (self.correct / float(self.total))
        return {'acc': acc}

    def reset(self):
        self.total = 0
        self.correct = 0
