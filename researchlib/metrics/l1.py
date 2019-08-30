from .matrix import *
import torch


class L1(Matrix):

    def __init__(self):
        super().__init__()
        self.total = 0
        self.value = 0

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup
            y_pred, y_true, y_true_res, lam = loss_input[0].cpu(
            ), loss_input[1].cpu(), loss_input[2].cpu(), loss_input[3]
            self.value += torch.abs(y_pred - (lam * y_true +
                                              (1 - lam) * y_true_res)).mean()
            self.total += 1
        else:
            y_pred, y_true = loss_input[0].detach().cpu(), loss_input[1].detach(
            ).cpu()
            self.value += torch.abs(y_pred - y_true).sum()
            self.total += y_true.size(0)

    def output(self):
        mae = (self.value / float(self.total))
        return {'mae': mae}

    def reset(self):
        self.total = 0
        self.value = 0
