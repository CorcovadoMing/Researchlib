from .matrix import *
import torch
import numpy as np


class MAAPE(Matrix):
    """ Compute the the Mean Arctangent Absolute Percentage Error (MAAPE).
    $ formula := mean_t(arctan(|(At-Ft)/At|)) $
    Range: 0 ≤ MAAPE < π/2, does not indicate bias, smaller is better.
    Notes: Represents the mean absolute error as a percentage of the observed values. Handles 0s in the observed data. This metric is not as biased as MAPE by under-over predictions.
    """
    def __init__(self, epsilon=1e-9):
        super().__init__()
        self.total = 0
        self.value = 0
        self.epsilon = epsilon

    def forward(self, loss_input):
        y_pred, y_true = loss_input[0].cpu(), loss_input[1].cpu()
        self.value += np.arctan(np.abs(y_true - y_pred) / (y_true+self.epsilon)).mean()
        self.total += 1

    def output(self):
        maape = (self.value / float(self.total))
        return {'maape': maape}

    def reset(self):
        self.total = 0
        self.value = 0
