from .matrix import *
from ..utils import *

import numpy as np
import matplotlib.pyplot as plt
import torch


class ConfusionMatrix(Matrix):
    def __init__(self, classes, plot=False):
        super().__init__()
        self.classes = classes
        self.m = np.zeros((classes, classes)).astype(np.int)
        self.plot = plot

    def forward(self, loss_input):
        y_pred, y_true = loss_input[0].cpu().argmax(-1), loss_input[1].cpu()
        for (i, j) in zip(y_pred, y_true):
            self.m[i, j] += 1

    def output(self):
        if self.plot:
            score = np.diag(self.m).sum() / float(self.m.sum())
            plt.figure(figsize=(7, 7))
            plt.imshow(self.m, cmap='gray')
            for i in range(self.classes):
                for j in range(self.classes):
                    plt.text(j,
                             i,
                             self.m[i, j],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='gray')
            plt.ylabel('Predicted')
            plt.yticks(range(self.classes))
            plt.xlabel('Actual')
            plt.xticks(range(self.classes))
            all_sample_title = 'Accuracy Score: {0}'.format(score)
            plt.title(all_sample_title)
        else:
            print(self.m)
        return {}

    def reset(self):
        self.m = np.zeros((self.classes, self.classes)).astype(np.int)


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
            y_pred, y_true = loss_input[0].detach(), loss_input[1].detach()
            _, predicted = torch.max(y_pred, 1)
            self.correct += predicted.eq(y_true.long()).sum().float()
        self.total += predicted.view(-1).size(0)

    def output(self):
        acc = (self.correct / float(self.total))
        return {'acc': acc}

    def reset(self):
        self.total = 0
        self.correct = 0


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
            y_pred, y_true = loss_input[0].detach(), loss_input[1].detach()
            predicted = (y_pred > 0.5).float()
            self.total += y_true.size(0)
            self.correct += predicted.eq(y_true).sum().float()

    def output(self):
        acc = (self.correct / float(self.total))
        return {'acc': acc}

    def reset(self):
        self.total = 0
        self.correct = 0
