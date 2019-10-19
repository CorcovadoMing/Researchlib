from .matrix import Matrix
import torch


class MetaAcc(Matrix):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0

    def forward(self, loss_input):
        y_pred, y_true = loss_input[0].detach(), loss_input[1].detach()
        _, y_true = torch.max(y_true, -1)
        y_true, _ = torch.max(y_true, -1)
        y_pred = y_pred.sum(1)
        _, y_pred = torch.max(y_pred, -1)
        self.correct += y_pred.eq(y_true).sum().float()
        self.total += y_pred.size(0)

    def output(self):
        meta_acc = (self.correct / float(self.total))
        return {'meta_acc': meta_acc}

    def reset(self):
        self.total = 0
        self.correct = 0
