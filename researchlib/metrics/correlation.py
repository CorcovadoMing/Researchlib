from scipy.stats import spearmanr
from .matrix import *
import numpy as np

class Correlation(Matrix):
    def __init__(self, difference=True, categorical=False):
        super().__init__()
        self.pred = None
        self.true = None
        self.categorical = categorical
        self.difference = difference
        
    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup (INCOMPLETE!!)
            pass
        else:
            y_pred, y_true = loss_input[0], loss_input[1]
            y_pred, y_true = y_pred.detach(), y_true.detach()
            if self.categorical:
                y_pred = y_pred.argmax(-1)
            if self.difference:
                pad = (y_true - y_pred).cpu().numpy()
            else:
                pad = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            if self.pred is None:
                self.pred = pad
            else:
                self.pred = np.concatenate((self.pred, pad))
            if self.true is None:
                self.true = y_true
            else:
                self.true = np.concatenate((self.true, y_true))
                
    def output(self):
        correlation = spearmanr(self.pred, self.true)[0]
        return {'correlation': correlation}
    
    def reset(self):
        self.pred = None
        self.true = None