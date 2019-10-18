from .matrix import *
from ..utils import *
import numpy as np
import matplotlib.pyplot as plt
import torch


class ConfusionMatrix(Matrix):
    def __init__(self, classes, plot = False):
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
            plt.figure(figsize = (7, 7))
            plt.imshow(self.m, cmap = 'gray')
            for i in range(self.classes):
                for j in range(self.classes):
                    plt.text(
                        j,
                        i,
                        self.m[i, j],
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        color = 'gray'
                    )
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
