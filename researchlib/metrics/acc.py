from .matrix import Matrix
import torch


class Acc(Matrix):

    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup
            y_pred, y_true, y_true_res, lam = loss_input[0].detach(
            ), loss_input[1].detach(), loss_input[2].detach(), loss_input[3]
            _, predicted = torch.max(y_pred, 1)
            self.correct += (
                lam * predicted.eq(y_true.long()).sum().float() +
                (1 - lam) * predicted.eq(y_true_res.long()).sum().float())
        else:
            y_pred, y_true = loss_input[0].detach(), loss_input[1].detach().squeeze()
            _, predicted = torch.max(y_pred, 1)
            self.correct += predicted.eq(y_true.long()).sum().float()
        self.total += predicted.view(-1).size(0)

    def output(self):
        acc = (self.correct / float(self.total))
        return {'acc': acc}

    def reset(self):
        self.total = 0
        self.correct = 0
