from .matrix import *
import torch


class L2(Matrix):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.value = 0

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup
            y_pred, y_true, y_true_res, lam = loss_input[0].cpu(), loss_input[1].cpu(
            ), loss_input[2].cpu(), loss_input[3]
            self.value += (y_pred - (lam * y_true + (1 - lam) * y_true_res)).pow(2).mean()
            self.total += 1
        else:
            y_pred, y_true = loss_input[0].detach().cpu(), loss_input[1].detach().cpu()
            self.value += (y_pred - y_true).pow(2).sum()
            self.total += y_true.size(0)

    def output(self):
        mse = (self.value / float(self.total))
        return {'mse': mse}

    def reset(self):
        self.total = 0
        self.value = 0
