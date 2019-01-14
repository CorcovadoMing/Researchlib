from .matrix import *
import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix(Matrix):
    def __init__(self, classes, plot=False):
        super().__init__()
        self.classes = classes
        self.m = np.zeros((classes, classes)).astype(np.int)
        self.plot = plot
        
    def forward(self, y_pred, y_true):
        y_pred, y_true = self.prepare(y_pred, y_true)
        for (i, j) in zip(y_pred, y_true):
            self.m[i, j] += 1
        
    def output(self):
        if self.plot:
            score = np.diag(self.m).sum() / float(self.m.sum())
            plt.figure(figsize=(7, 7))
            plt.imshow(self.m, cmap='gray')
            for i in range(self.classes):
                for j in range(self.classes):
                    plt.text(j, i, self.m[i, j], horizontalalignment='center', verticalalignment='center', color='gray')
            plt.ylabel('Actual');
            plt.yticks(range(self.classes))
            plt.xlabel('Predicted');
            plt.xticks(range(self.classes))
            all_sample_title = 'Accuracy Score: {0}'.format(score)
            plt.title(all_sample_title);
        else:    
            print(self.m)
        
    def reset(self):
        self.m = np.zeros((self.classes, self.classes)).astype(np.int)


class Acc(Matrix):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0
        
    def forward(self, y_pred, y_true):
        y_pred, y_true = self.prepare(y_pred, y_true)
        for (i, j) in zip(y_pred, y_true):
            if i == j:
                self.correct += 1
            self.total += 1
        
    def output(self):
        print(self.correct / self.total)
        
    def reset(self):
        self.total = 0
        self.correct = 0
    
    